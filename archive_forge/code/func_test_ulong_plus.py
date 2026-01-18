import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_ulong_plus(self):
    self._dll.tf_bL.restype = c_ulong
    self._dll.tf_bL.argtypes = (c_char, c_ulong)
    self.assertEqual(self._dll.tf_bL(b' ', 4294967295), 1431655765)
    self.assertEqual(self.U(), 4294967295)