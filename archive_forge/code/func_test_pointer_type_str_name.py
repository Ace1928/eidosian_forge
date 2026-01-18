import unittest, sys
from ctypes import *
import _ctypes_test
def test_pointer_type_str_name(self):
    large_string = 'T' * 2 ** 25
    P = POINTER(large_string)
    self.assertTrue(P)
    from ctypes import _pointer_type_cache
    del _pointer_type_cache[id(P)]