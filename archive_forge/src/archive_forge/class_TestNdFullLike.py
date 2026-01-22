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
class TestNdFullLike(ConstructorLikeBaseTest, TestCase):

    def check_result_value(self, ret, expected):
        np.testing.assert_equal(ret, expected)

    def test_like(self):

        def func(arr):
            return np.full_like(arr, 3.5)
        self.check_like(func, np.float64)

    @unittest.expectedFailure
    def test_like_structured(self):
        dtype = np.dtype([('a', np.int16), ('b', np.float32)])

        def func(arr):
            return np.full_like(arr, 4.5)
        self.check_like(func, dtype)

    def test_like_dtype(self):

        def func(arr):
            return np.full_like(arr, 4.5, np.bool_)
        self.check_like(func, np.float64)

    def test_like_dtype_instance(self):
        dtype = np.dtype('bool')

        def func(arr):
            return np.full_like(arr, 4.5, dtype)
        self.check_like(func, np.float64)

    def test_like_dtype_kwarg(self):

        def func(arr):
            return np.full_like(arr, 4.5, dtype=np.bool_)
        self.check_like(func, np.float64)

    def test_like_dtype_str_kwarg(self):

        def func(arr):
            return np.full_like(arr, 4.5, 'bool_')
        self.check_like(func, np.float64)

    def test_like_dtype_str_kwarg_alternative_spelling(self):

        def func(arr):
            return np.full_like(arr, 4.5, dtype='?')
        self.check_like(func, np.float64)

    def test_like_dtype_non_const_str_kwarg(self):

        @njit
        def func(arr, fv, dt):
            return np.full_like(arr, fv, dt)
        with self.assertRaises(TypingError) as raises:
            func(np.ones(3), 4.5, 'int32')
        excstr = str(raises.exception)
        msg = 'If np.full_like dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)

    def test_like_dtype_invalid_str(self):

        @njit
        def func(arr, fv):
            return np.full_like(arr, fv, 'ABCDEF')
        with self.assertRaises(TypingError) as raises:
            func(np.ones(4), 3.4)
        excstr = str(raises.exception)
        self.assertIn("Invalid NumPy dtype specified: 'ABCDEF'", excstr)