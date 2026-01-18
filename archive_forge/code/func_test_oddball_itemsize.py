import sys
import ctypes
from ctypes import *
import unittest
def test_oddball_itemsize(self):
    for size in [3, 5, 6, 7, 9]:
        a = Array((1,), 'i', size)
        ct = a._ctype
        self.assertTrue(issubclass(ct, ctypes.Array))
        self.assertEqual(sizeof(ct), size)