from ctypes import *
import sys, platform, struct
def NSMakeRect(x, y, w, h):
    return NSRect(NSPoint(x, y), NSSize(w, h))