import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_poly_polyval_exception(self):
    cfunc2 = njit(polyval2)
    cfunc3T = njit(polyval3T)
    cfunc3F = njit(polyval3F)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc2(3, 'abc')
    self.assertIn('The argument "c" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc2('abc', 3)
    self.assertIn('The argument "x" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc2('abc', 'def')
    self.assertIn('The argument "x" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc3T(3, 'abc')
    self.assertIn('The argument "c" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc3T('abc', 3)
    self.assertIn('The argument "x" must be array-like', str(raises.exception))

    @njit
    def polyval3(x, c, tensor):
        res = poly.polyval(x, c, tensor)
        return res
    with self.assertRaises(TypingError) as raises:
        polyval3(3, 3, 'abc')
    self.assertIn('The argument "tensor" must be boolean', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc3F('abc', 'def')
    self.assertIn('The argument "x" must be array-like', str(raises.exception))