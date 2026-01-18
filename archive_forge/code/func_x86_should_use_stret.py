import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def x86_should_use_stret(restype):
    """Try to figure out when a return type will be passed on stack."""
    if type(restype) != type(Structure):
        return False
    if not __LP64__ and sizeof(restype) <= 8:
        return False
    if __LP64__ and sizeof(restype) <= 16:
        return False
    return True