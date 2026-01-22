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
class TestNpyEmptyKeyword(TestCase):

    def _test_with_dtype_kw(self, dtype):

        def pyfunc(shape):
            return np.empty(shape, dtype=dtype)
        shapes = [1, 5, 9]
        cfunc = nrtjit(pyfunc)
        for s in shapes:
            expected = pyfunc(s)
            got = cfunc(s)
            self.assertEqual(expected.dtype, got.dtype)
            self.assertEqual(expected.shape, got.shape)

    def test_with_dtype_kws(self):
        for dtype in [np.int32, np.float32, np.complex64, np.dtype('complex64')]:
            self._test_with_dtype_kw(dtype)

    def _test_with_shape_and_dtype_kw(self, dtype):

        def pyfunc(shape):
            return np.empty(shape=shape, dtype=dtype)
        shapes = [1, 5, 9]
        cfunc = nrtjit(pyfunc)
        for s in shapes:
            expected = pyfunc(s)
            got = cfunc(s)
            self.assertEqual(expected.dtype, got.dtype)
            self.assertEqual(expected.shape, got.shape)

    def test_with_shape_and_dtype_kws(self):
        for dtype in [np.int32, np.float32, np.complex64, np.dtype('complex64')]:
            self._test_with_shape_and_dtype_kw(dtype)

    def test_empty_no_args(self):

        def pyfunc():
            return np.empty()
        cfunc = nrtjit(pyfunc)
        with self.assertRaises(TypingError):
            cfunc()