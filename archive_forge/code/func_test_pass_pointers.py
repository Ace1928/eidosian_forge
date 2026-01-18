import unittest, sys
from ctypes import *
import _ctypes_test
def test_pass_pointers(self):
    dll = CDLL(_ctypes_test.__file__)
    func = dll._testfunc_p_p
    if sizeof(c_longlong) == sizeof(c_void_p):
        func.restype = c_longlong
    else:
        func.restype = c_long
    i = c_int(12345678)
    address = func(byref(i))
    self.assertEqual(c_int.from_address(address).value, 12345678)
    func.restype = POINTER(c_int)
    res = func(pointer(i))
    self.assertEqual(res.contents.value, 12345678)
    self.assertEqual(res[0], 12345678)