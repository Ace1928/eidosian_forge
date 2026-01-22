from ctypes import *
from ctypes import util
from .runtime import send_message, ObjCInstance
from .cocoatypes import *
def CFSTR(string):
    return cf.CFStringCreateWithCString(None, string.encode('utf8'), kCFStringEncodingUTF8)