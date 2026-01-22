from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
class X2(object, _Pointer):
    pass