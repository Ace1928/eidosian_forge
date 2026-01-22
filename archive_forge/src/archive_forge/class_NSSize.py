from ctypes import *
import sys, platform, struct
class NSSize(Structure):
    _fields_ = [('width', CGFloat), ('height', CGFloat)]