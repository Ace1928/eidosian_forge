import builtins
import unittest
from numbers import Number
from functools import wraps
import numpy as np
from llvmlite import ir
import numba
from numba import njit, typeof, objmode
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import (intrinsic, lower_builtin, overload_classmethod,
from numba.np import numpy_support
from numba.tests.support import TestCase, MemoryLeakMixin
class TestNdarraySubclasses(MemoryLeakMixin, TestCase):

    def test_myarray_return(self):
        """This tests the path to `MyArrayType.box_type`
        """

        @njit
        def foo(a):
            return a + 1
        buf = np.arange(4)
        a = MyArray(buf.shape, buf.dtype, buf)
        expected = foo.py_func(a)
        got = foo(a)
        self.assertIsInstance(got, MyArray)
        self.assertIs(type(expected), type(got))
        self.assertPreciseEqual(expected, got)

    def test_myarray_passthru(self):

        @njit
        def foo(a):
            return a
        buf = np.arange(4)
        a = MyArray(buf.shape, buf.dtype, buf)
        expected = foo.py_func(a)
        got = foo(a)
        self.assertIsInstance(got, MyArray)
        self.assertIs(type(expected), type(got))
        self.assertPreciseEqual(expected, got)

    def test_myarray_convert(self):

        @njit
        def foo(buf):
            return MyArray(buf.shape, buf.dtype, buf)
        buf = np.arange(4)
        expected = foo.py_func(buf)
        got = foo(buf)
        self.assertIsInstance(got, MyArray)
        self.assertIs(type(expected), type(got))
        self.assertPreciseEqual(expected, got)

    def test_myarray_asarray_non_jit(self):

        def foo(buf):
            converted = MyArray(buf.shape, buf.dtype, buf)
            return np.asarray(converted) + buf
        buf = np.arange(4)
        got = foo(buf)
        self.assertIs(type(got), np.ndarray)
        self.assertPreciseEqual(got, buf + buf)

    @unittest.expectedFailure
    def test_myarray_asarray(self):
        self.disable_leak_check()

        @njit
        def foo(buf):
            converted = MyArray(buf.shape, buf.dtype, buf)
            return np.asarray(converted)
        buf = np.arange(4)
        got = foo(buf)
        self.assertIs(type(got), np.ndarray)

    def test_myarray_ufunc_unsupported(self):

        @njit
        def foo(buf):
            converted = MyArray(buf.shape, buf.dtype, buf)
            return converted + converted
        buf = np.arange(4, dtype=np.float32)
        with self.assertRaises(TypingError) as raises:
            foo(buf)
        msg = ('No implementation of function', 'add(MyArray(1, float32, C), MyArray(1, float32, C))')
        for m in msg:
            self.assertIn(m, str(raises.exception))

    @use_logger
    def test_myarray_allocator_override(self):
        """
        Checks that our custom allocator is used
        """

        @njit
        def foo(a):
            b = a + np.arange(a.size, dtype=np.float64)
            c = a + 1j
            return (b, c)
        buf = np.arange(4, dtype=np.float64)
        a = MyArray(buf.shape, buf.dtype, buf)
        expected = foo.py_func(a)
        got = foo(a)
        self.assertPreciseEqual(got, expected)
        logged_lines = _logger
        targetctx = cpu_target.target_context
        nb_dtype = typeof(buf.dtype)
        align = targetctx.get_preferred_array_alignment(nb_dtype)
        self.assertEqual(logged_lines, [('LOG _ol_array_allocate', expected[0].nbytes, align), ('LOG _ol_array_allocate', expected[1].nbytes, align)])