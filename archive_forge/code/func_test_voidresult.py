from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_voidresult(self):
    f = dll._testfunc_v
    f.restype = None
    f.argtypes = [c_int, c_int, POINTER(c_int)]
    result = c_int()
    self.assertEqual(None, f(1, 2, byref(result)))
    self.assertEqual(result.value, 3)