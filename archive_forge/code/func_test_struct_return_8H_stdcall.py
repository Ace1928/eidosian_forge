from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
@unittest.skipUnless(sys.platform == 'win32', 'Windows-specific test')
def test_struct_return_8H_stdcall(self):

    class S8I(Structure):
        _fields_ = [('a', c_int), ('b', c_int), ('c', c_int), ('d', c_int), ('e', c_int), ('f', c_int), ('g', c_int), ('h', c_int)]
    windll.s_ret_8i_func.restype = S8I
    windll.s_ret_8i_func.argtypes = [S8I]
    inp = S8I(9, 8, 7, 6, 5, 4, 3, 2)
    s8i = windll.s_ret_8i_func(inp)
    self.assertEqual((s8i.a, s8i.b, s8i.c, s8i.d, s8i.e, s8i.f, s8i.g, s8i.h), (9 * 2, 8 * 3, 7 * 4, 6 * 5, 5 * 6, 4 * 7, 3 * 8, 2 * 9))