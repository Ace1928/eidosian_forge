import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
class TestNpConcatenate(MemoryLeakMixin, TestCase):
    """
    Tests for np.concatenate().
    """

    def _3d_arrays(self):
        a = np.arange(24).reshape((4, 3, 2))
        b = a + 10
        c = (b + 10).copy(order='F')
        d = (c + 10)[::-1]
        e = (d + 10)[..., ::-1]
        return (a, b, c, d, e)

    @contextlib.contextmanager
    def assert_invalid_sizes_over_dim(self, axis):
        with self.assertRaises(ValueError) as raises:
            yield
        self.assertIn('input sizes over dimension %d do not match' % axis, str(raises.exception))

    def test_3d(self):
        pyfunc = np_concatenate2
        cfunc = nrtjit(pyfunc)

        def check(a, b, c, axis):
            for ax in (axis, -3 + axis):
                expected = pyfunc(a, b, c, axis=ax)
                got = cfunc(a, b, c, axis=ax)
                self.assertPreciseEqual(got, expected)

        def check_all_axes(a, b, c):
            for axis in range(3):
                check(a, b, c, axis)
        a, b, c, d, e = self._3d_arrays()
        check_all_axes(a, b, b)
        check_all_axes(a, b, c)
        check_all_axes(a.T, b.T, a.T)
        check_all_axes(a.T, b.T, c.T)
        check_all_axes(a.T, b.T, d.T)
        check_all_axes(d.T, e.T, d.T)
        check(a[1:], b, c[::-1], axis=0)
        check(a, b[:, 1:], c, axis=1)
        check(a, b, c[:, :, 1:], axis=2)
        check_all_axes(a, b.astype(np.float64), b)
        self.disable_leak_check()
        for axis in (1, 2, -2, -1):
            with self.assert_invalid_sizes_over_dim(0):
                cfunc(a[1:], b, b, axis)
        for axis in (0, 2, -3, -1):
            with self.assert_invalid_sizes_over_dim(1):
                cfunc(a, b[:, 1:], b, axis)

    def test_3d_no_axis(self):
        pyfunc = np_concatenate1
        cfunc = nrtjit(pyfunc)

        def check(a, b, c):
            expected = pyfunc(a, b, c)
            got = cfunc(a, b, c)
            self.assertPreciseEqual(got, expected)
        a, b, c, d, e = self._3d_arrays()
        check(a, b, b)
        check(a, b, c)
        check(a.T, b.T, a.T)
        check(a.T, b.T, c.T)
        check(a.T, b.T, d.T)
        check(d.T, e.T, d.T)
        check(a[1:], b, c[::-1])
        self.disable_leak_check()
        with self.assert_invalid_sizes_over_dim(1):
            cfunc(a, b[:, 1:], b)

    def test_typing_errors(self):
        pyfunc = np_concatenate1
        cfunc = nrtjit(pyfunc)
        a = np.arange(15)
        b = a.reshape((3, 5))
        c = a.astype(np.dtype([('x', np.int8)]))
        d = np.array(42)
        with self.assertTypingError() as raises:
            cfunc(a, b, b)
        self.assertIn('all the input arrays must have same number of dimensions', str(raises.exception))
        with self.assertTypingError() as raises:
            cfunc(a, c, c)
        self.assertIn('input arrays must have compatible dtypes', str(raises.exception))
        with self.assertTypingError() as raises:
            cfunc(d, d, d)
        self.assertIn('zero-dimensional arrays cannot be concatenated', str(raises.exception))
        with self.assertTypingError() as raises:
            cfunc(c, 1, c)
        self.assertIn('expecting a non-empty tuple of arrays', str(raises.exception))