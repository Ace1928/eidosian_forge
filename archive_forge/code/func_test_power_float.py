import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def test_power_float(self):
    """
        Test 0 ** -1 and 2 ** <big number>.
        """
    f = vectorize(nopython=True)(power)
    a = np.array([5.0, 0.0, 2.0, 8.0])
    b = np.array([1.0, -1.0, 1e+20, 4.0])
    expected = np.array([5.0, float('inf'), float('inf'), 4096.0])
    with self.check_warnings(['divide by zero encountered', 'overflow encountered']):
        res = f(a, b)
        self.assertPreciseEqual(res, expected)