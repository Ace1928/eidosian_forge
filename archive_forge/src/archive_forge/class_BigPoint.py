import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
class BigPoint(BigEndianStructure):
    __slots__ = ()
    _fields_ = [('x', c_int), ('y', c_int)]