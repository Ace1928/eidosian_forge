import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def test_power_integer(self):
    """
        Test 0 ** -1.
        Note 2 ** <big number> returns an undefined value (depending
        on the algorithm).
        """
    dtype = np.int64
    f = vectorize(['int64(int64, int64)'], nopython=True)(power)
    a = np.array([5, 0, 6], dtype=dtype)
    b = np.array([1, -1, 2], dtype=dtype)
    expected = np.array([5, -2 ** 63, 36], dtype=dtype)
    with self.check_warnings([]):
        res = f(a, b)
        self.assertPreciseEqual(res, expected)