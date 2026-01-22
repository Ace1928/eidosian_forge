import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
class TestBasicSubtyping(TestCase):

    def test_basic(self):
        """
        Test that a dispatcher object *with* a pre-compiled overload
        can be used as input to another function with locked-down signature
        """
        a = 1

        @njit
        def foo(x):
            return x + 1
        foo(a)
        int_int_fc = types.FunctionType(types.int64(types.int64))

        @njit(types.int64(int_int_fc))
        def bar(fc):
            return fc(a)
        self.assertEqual(bar(foo), foo(a))

    def test_basic2(self):
        """
        Test that a dispatcher object *without* a pre-compiled overload
        can be used as input to another function with locked-down signature
        """
        a = 1

        @njit
        def foo(x):
            return x + 1
        int_int_fc = types.FunctionType(types.int64(types.int64))

        @njit(types.int64(int_int_fc))
        def bar(fc):
            return fc(a)
        self.assertEqual(bar(foo), foo(a))

    def test_basic3(self):
        """
        Test that a dispatcher object *without* a pre-compiled overload
        can be used as input to another function with locked-down signature and
        that it behaves as a truly generic function (foo1 does not get locked)
        """
        a = 1

        @njit
        def foo1(x):
            return x + 1

        @njit
        def foo2(x):
            return x + 2
        int_int_fc = types.FunctionType(types.int64(types.int64))

        @njit(types.int64(int_int_fc))
        def bar(fc):
            return fc(a)
        self.assertEqual(bar(foo1) + 1, bar(foo2))

    def test_basic4(self):
        """
        Test that a dispatcher object can be used as input to another
         function with signature as part of a tuple
        """
        a = 1

        @njit
        def foo1(x):
            return x + 1

        @njit
        def foo2(x):
            return x + 2
        tup = (foo1, foo2)
        int_int_fc = types.FunctionType(types.int64(types.int64))

        @njit(types.int64(types.UniTuple(int_int_fc, 2)))
        def bar(fcs):
            x = 0
            for i in range(2):
                x += fcs[i](a)
            return x
        self.assertEqual(bar(tup), foo1(a) + foo2(a))

    def test_basic5(self):
        a = 1

        @njit
        def foo1(x):
            return x + 1

        @njit
        def foo2(x):
            return x + 2

        @njit
        def bar1(x):
            return x / 10

        @njit
        def bar2(x):
            return x / 1000
        tup = (foo1, foo2)
        tup_bar = (bar1, bar2)
        int_int_fc = types.FunctionType(types.int64(types.int64))
        flt_flt_fc = types.FunctionType(types.float64(types.float64))

        @njit((types.UniTuple(int_int_fc, 2), types.UniTuple(flt_flt_fc, 2)))
        def bar(fcs, ffs):
            x = 0
            for i in range(2):
                x += fcs[i](a)
            for fn in ffs:
                x += fn(a)
            return x
        got = bar(tup, tup_bar)
        expected = foo1(a) + foo2(a) + bar1(a) + bar2(a)
        self.assertEqual(got, expected)