from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_shorts(self):
    f = dll._testfunc_callback_i_if
    args = []
    expected = [262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

    def callback(v):
        args.append(v)
        return v
    CallBack = CFUNCTYPE(c_int, c_int)
    cb = CallBack(callback)
    f(2 ** 18, cb)
    self.assertEqual(args, expected)