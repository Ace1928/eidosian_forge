import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
class TestFloatingPointExceptions(TestCase, CheckWarningsMixin):
    """
    Test floating-point exceptions inside ufuncs.

    Note the warnings emitted by Numpy reflect IEEE-754 semantics.
    """

    def check_truediv_real(self, dtype):
        """
        Test 1 / 0 and 0 / 0.
        """
        f = vectorize(nopython=True)(truediv)
        a = np.array([5.0, 6.0, 0.0, 8.0], dtype=dtype)
        b = np.array([1.0, 0.0, 0.0, 4.0], dtype=dtype)
        expected = np.array([5.0, float('inf'), float('nan'), 2.0])
        with self.check_warnings(['divide by zero encountered', 'invalid value encountered']):
            res = f(a, b)
            self.assertPreciseEqual(res, expected)

    def test_truediv_float(self):
        self.check_truediv_real(np.float64)

    def test_truediv_integer(self):
        self.check_truediv_real(np.int32)

    def check_divmod_float(self, pyfunc, values, messages):
        """
        Test 1 // 0 and 0 // 0.
        """
        f = vectorize(nopython=True)(pyfunc)
        a = np.array([5.0, 6.0, 0.0, 9.0])
        b = np.array([1.0, 0.0, 0.0, 4.0])
        expected = np.array(values)
        with self.check_warnings(messages):
            res = f(a, b)
            self.assertPreciseEqual(res, expected)

    def test_floordiv_float(self):
        self.check_divmod_float(floordiv, [5.0, float('inf'), float('nan'), 2.0], ['divide by zero encountered', 'invalid value encountered'])

    @skip_m1_fenv_errors
    def test_remainder_float(self):
        self.check_divmod_float(remainder, [0.0, float('nan'), float('nan'), 1.0], ['invalid value encountered'])

    def check_divmod_int(self, pyfunc, values):
        """
        Test 1 % 0 and 0 % 0.
        """
        f = vectorize(nopython=True)(pyfunc)
        a = np.array([5, 6, 0, 9])
        b = np.array([1, 0, 0, 4])
        expected = np.array(values)
        with self.check_warnings([]):
            res = f(a, b)
            self.assertPreciseEqual(res, expected)

    def test_floordiv_int(self):
        self.check_divmod_int(floordiv, [5, 0, 0, 2])

    def test_remainder_int(self):
        self.check_divmod_int(remainder, [0, 0, 0, 1])

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