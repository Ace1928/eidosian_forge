import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
class NestedStructure(nested):
    _fields_ = [('x', c_uint32), ('y', c_uint32)]