import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_void(self):
    self._dll.tv_i.restype = None
    self._dll.tv_i.argtypes = (c_int,)
    self.assertEqual(self._dll.tv_i(42), None)
    self.assertEqual(self.S(), 42)
    self.assertEqual(self._dll.tv_i(-42), None)
    self.assertEqual(self.S(), -42)