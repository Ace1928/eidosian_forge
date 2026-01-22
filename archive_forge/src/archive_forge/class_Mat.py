import sys
import cv2 as cv
class Mat:

    def __new__(self):
        return cv.GArrayT(cv.gapi.CV_MAT)