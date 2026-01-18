import sys
import ctypes
from ctypes import *
import unittest
def test_typekind(self):
    a = Array((1,), 'i', 4)
    self.assertTrue(a._ctype is c_int32)
    self.assertTrue(a._ctype_p is POINTER(c_int32))
    a = Array((1,), 'u', 4)
    self.assertTrue(a._ctype is c_uint32)
    self.assertTrue(a._ctype_p is POINTER(c_uint32))
    a = Array((1,), 'f', 4)
    ct = a._ctype
    self.assertTrue(issubclass(ct, ctypes.Array))
    self.assertEqual(sizeof(ct), 4)