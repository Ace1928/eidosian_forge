from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
@unittest.skipUnless(sys.platform == 'win32', 'Windows-specific test')
def test_struct_return_2H_stdcall(self):

    class S2H(Structure):
        _fields_ = [('x', c_short), ('y', c_short)]
    windll.s_ret_2h_func.restype = S2H
    windll.s_ret_2h_func.argtypes = [S2H]
    s2h = windll.s_ret_2h_func(S2H(99, 88))
    self.assertEqual((s2h.x, s2h.y), (99 * 2, 88 * 3))