from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
class RECT(Structure):
    _fields_ = [('left', c_int), ('top', c_int), ('right', c_int), ('bottom', c_int)]