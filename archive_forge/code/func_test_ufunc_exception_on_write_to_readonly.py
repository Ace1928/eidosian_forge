import numpy as np
from numba import float32, jit, njit
from numba.np.ufunc import Vectorize
from numba.core.errors import TypingError
from numba.tests.support import TestCase
import unittest
def test_ufunc_exception_on_write_to_readonly(self):
    z = np.ones(10)
    z.flags.writeable = False
    tests = []
    expect = "ufunc 'sin' called with an explicit output that is read-only"
    tests.append((jit(nopython=True), TypingError, expect))
    tests.append((jit(forceobj=True), ValueError, 'output array is read-only'))
    for dec, exc, msg in tests:

        def test(x):
            a = np.ones(x.shape, x.dtype)
            np.sin(a, x)
        with self.assertRaises(exc) as raises:
            dec(test)(z)
        self.assertIn(msg, str(raises.exception))