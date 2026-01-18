import unittest
import ctypes
from ctypes.test import need_symbol
import _ctypes_test
def test_wcslen(self):
    dll = ctypes.CDLL(_ctypes_test.__file__)
    wcslen = dll.my_wcslen
    wcslen.argtypes = [ctypes.c_wchar_p]
    self.assertEqual(wcslen('abc'), 3)
    self.assertEqual(wcslen('ab⁰'), 3)
    self.assertRaises(ctypes.ArgumentError, wcslen, b'ab\xe4')