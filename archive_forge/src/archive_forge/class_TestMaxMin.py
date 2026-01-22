from numba import njit
from numba.core import types
import unittest
class TestMaxMin(unittest.TestCase):

    def test_max3(self):
        pyfunc = domax3
        argtys = (types.int32, types.float32, types.double)
        cfunc = njit(argtys)(pyfunc)
        a = 1
        b = 2
        c = 3
        self.assertEqual(pyfunc(a, b, c), cfunc(a, b, c))

    def test_min3(self):
        pyfunc = domin3
        argtys = (types.int32, types.float32, types.double)
        cfunc = njit(argtys)(pyfunc)
        a = 1
        b = 2
        c = 3
        self.assertEqual(pyfunc(a, b, c), cfunc(a, b, c))