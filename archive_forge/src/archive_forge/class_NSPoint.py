from ctypes import *
import sys, platform, struct
class NSPoint(Structure):
    _fields_ = [('x', CGFloat), ('y', CGFloat)]