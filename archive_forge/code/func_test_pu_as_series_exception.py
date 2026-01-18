import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_pu_as_series_exception(self):
    cfunc1 = njit(polyasseries1)
    cfunc2 = njit(polyasseries2)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc1('abc')
    self.assertIn('The argument "alist" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc2('abc', True)
    self.assertIn('The argument "alist" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc2(np.arange(4), 'abc')
    self.assertIn('The argument "trim" must be boolean', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc1(([1, 2, 3], np.arange(16).reshape(4, 4)))
    self.assertIn('Coefficient array is not 1-d', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc1(np.arange(8).reshape((2, 2, 2)))
    self.assertIn('Coefficient array is not 1-d', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc1([np.array([[1, 2, 3], [1, 2, 3]])])
    self.assertIn('Coefficient array is not 1-d', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc1(np.array([[]], dtype=np.float64))
    self.assertIn('Coefficient array is empty', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc1(([1, 2, 3], np.array([], dtype=np.float64), np.array([1, 2, 1])))
    self.assertIn('Coefficient array is empty', str(raises.exception))