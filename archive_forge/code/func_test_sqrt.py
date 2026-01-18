import unittest
from ctypes import *
import _ctypes_test
def test_sqrt(self):
    lib.my_sqrt.argtypes = (c_double,)
    lib.my_sqrt.restype = c_double
    self.assertEqual(lib.my_sqrt(4.0), 2.0)
    import math
    self.assertEqual(lib.my_sqrt(2.0), math.sqrt(2.0))