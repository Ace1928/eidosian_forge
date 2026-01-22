from ctypes import *
import sys, platform, struct
class NSRect(Structure):
    _fields_ = [('origin', NSPoint), ('size', NSSize)]