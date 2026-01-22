import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
class TestStructure(parent):
    _fields_ = [('point', NestedStructure)]