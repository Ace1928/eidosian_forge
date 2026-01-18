import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_ubyte_plus(self):
    self._dll.tf_bB.restype = c_ubyte
    self._dll.tf_bB.argtypes = (c_byte, c_ubyte)
    self.assertEqual(self._dll.tf_bB(0, 255), 85)
    self.assertEqual(self.U(), 255)