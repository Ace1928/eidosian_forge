import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_short_plus(self):
    self._dll.tf_bh.restype = c_short
    self._dll.tf_bh.argtypes = (c_byte, c_short)
    self.assertEqual(self._dll.tf_bh(0, -32766), -10922)
    self.assertEqual(self.S(), -32766)