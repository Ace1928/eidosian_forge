from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_sf1651235(self):
    proto = CFUNCTYPE(c_int, RECT, POINT)

    def callback(*args):
        return 0
    callback = proto(callback)
    self.assertRaises(ArgumentError, lambda: callback((1, 2, 3, 4), POINT()))