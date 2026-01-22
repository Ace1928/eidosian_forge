import numpy
import cv2
import time
import pygame
class CameraMac(Camera):

    def __init__(self, device=0, size=(640, 480), mode='RGB', api_preference=None):
        if isinstance(device, int):
            _dev = device
        elif isinstance(device, str):
            _dev = list_cameras_darwin().index(device)
        else:
            raise TypeError('OpenCV-Mac backend can take device indices or names, ints or strings, not ', str(type(device)))
        super().__init__(_dev, size, mode, api_preference)