import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import math
from skimage.filters import threshold_sauvola, threshold_local
import imutils
import pandas as pd
import argparse
import sys
import os



def readimg(path):
    #converts rgb ro grayscale
    img = cv2.imread(path,0)
    return img

def denoise(img):
    #apply a gaussian Filter
    blur = cv2.GaussianBlur(img.copy(),(5,5),0)
    return blur

def denoise2(img):
    #apply a gaussian Filter
    median = cv2.medianBlur(img.copy(),5)
    return median

def binarize(img):
    tresh = threshold_sauvola(img.copy(), window_size=25)
    bin1 = (img > tresh)*255
    bin1=bin1.astype(np.uint8)
    return bin1

def otsu(img):
    bin1=cv2.threshold(img, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return bin1

def canny(img):
    edge = cv2.Canny(img.copy(), 75, 200)
    return edge

def detect_ctrs(img):
    #apply gaussian filter and canny twice for more precision
    blur=denoise(img.copy())
    blur=denoise(blur)
    ed1=canny(blur)
    ed1=canny(ed1)
    return ed1

def largest_cont(edg):
    cnts, hierarchy  = cv2.findContours(edg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #select contour with the biggest area
    cnt = max(cnts , key = cv2.contourArea)
    #perimeter approximation (True --> closed contour)
    peri = cv2.arcLength(cnt, True)
    #polygon approximation
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    return approx

def cont_area(edg):
    #max contour area
    cnts, hierarchy  = cv2.findContours(edg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts , key = cv2.contourArea)
    return cv2.contourArea(cnt)

def draw_cont(img,edg):
    #draw contours
    cnts, hierarchy  = cv2.findContours(edg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts , key = cv2.contourArea)
    return cv2.drawContours(img.copy(), [cnt], -1, (0,255,0), 3)

def correct(cont,img):
    lig=cont.shape[0]
    cont=cont.reshape(lig,2)
    rect = np.zeros((4,2), dtype="float32")

    #quadrilateral estimation
    s = np.sum(cont, axis=1)
    rect[0] = cont[np.argmin(s)]
    rect[2] = cont[np.argmax(s)]

    diff = np.diff(cont, axis=1)
    rect[1] = cont[np.argmin(diff)]
    rect[3] = cont[np.argmax(diff)]

    (A, B, C, D) = rect

    #quadrilateral max(hauteur,largeur)
    widthA = np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2 )
    widthB = np.sqrt((D[0] - C[0])**2 + (D[1] - C[1])**2 )
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((A[0] - D[0])**2 + (A[1] - D[1])**2 )
    heightB = np.sqrt((B[0] - C[0])**2 + (B[1] - C[1])**2 )
    maxHeight = max(int(heightA), int(heightB))

    #reference quadrilateral
    dst = np.array([
    [0,0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]], dtype="float32")

    #transformation matrix (original quad --> reference quad)
    BansformMaBix = cv2.getPerspectiveTransform(rect, dst)
    #affine transformation
    scan = cv2.warpPerspective(img.copy(), BansformMaBix, (maxWidth, maxHeight),borderMode=cv2.BORDER_REPLICATE)

    return scan


def ratio(img):
    length=img.shape[0]
    if(length>4000):
        ratio=7.5
    elif(length>1500):
        ratio=5.5
    elif(length>870):
        ratio=4
    else :
        ratio=1
    return ratio

def create_arg_parser():
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('inputDirectory',
                    help='Path to the input directory.')
    parser.add_argument('--outputDirectory',
                    help='Path to the output that contains the resumes.')
    return parser


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if not os.path.exists(parsed_args.inputDirectory):
       print("File does not exist")
    else:
        path=parsed_args.inputDirectory
        # real image :
        rgb=cv2.imread(path)
        img=readimg(path)
        length=img.shape[0]
        ratio_v=ratio(img)

        div=int(length/ratio_v)
        #resize original image (drop height)
        im2 = imutils.resize(img, height = div)
        #rgb=imutils.resize(rgb, height = div)

        #find all contours
        edges=detect_ctrs(im2)
        #get the biggest one(main contour)
        main_cont=largest_cont(edges)

        if(len(main_cont)==0):
            #no contours only text
            res=im2
        else :
            area=cont_area(edges)
            #draw contous: visualisation--> draw=draw_cont(rgb,edges)

            if(area>400):
                #perspective correction
                res=correct(main_cont*ratio_v,img)


        #apply median filter
        res=denoise2(res)
        #binarize using sauvola or otsu :
        bin1=binarize(res)
        #bin1=otsu(tmp)
        l,c=bin1.shape
        #create a white rectangle 1px
        nbin=np.ones([l+2,c+2])*255
        nbin[1:-1,1:-1]=bin1
        #convert type to uint8
        nbin = nbin.astype(np.uint8)


        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(nbin,'gray')
        plt.show()
