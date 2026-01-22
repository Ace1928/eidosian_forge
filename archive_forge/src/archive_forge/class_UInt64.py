import sys
import cv2 as cv
class UInt64:

    def __new__(self):
        return cv.GArrayT(cv.gapi.CV_UINT64)