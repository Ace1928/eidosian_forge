import copy
import itertools
import math
import random
import sys
import unittest
import numpy as np
from numba import jit, njit
from numba.core import utils, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.misc.quicksort import make_py_quicksort, make_jit_quicksort
from numba.misc.mergesort import make_jit_mergesort
from numba.misc.timsort import make_py_timsort, make_jit_timsort, MergeRun
class TestSortSlashSortedWithKey(MemoryLeakMixin, TestCase):

    def test_01(self):
        a = [3, 1, 4, 1, 5, 9]

        @njit
        def external_key(z):
            return 1.0 / z

        @njit
        def foo(x, key=None):
            new_x = x[:]
            new_x.sort(key=key)
            return (sorted(x[:], key=key), new_x)
        self.assertPreciseEqual(foo(a[:]), foo.py_func(a[:]))
        self.assertPreciseEqual(foo(a[:], external_key), foo.py_func(a[:], external_key))

    def test_02(self):
        a = [3, 1, 4, 1, 5, 9]

        @njit
        def foo(x):

            def closure_key(z):
                return 1.0 / z
            new_x = x[:]
            new_x.sort(key=closure_key)
            return (sorted(x[:], key=closure_key), new_x)
        self.assertPreciseEqual(foo(a[:]), foo.py_func(a[:]))

    def test_03(self):
        a = [3, 1, 4, 1, 5, 9]

        def gen(compiler):

            @compiler
            def bar(x, func):
                new_x = x[:]
                new_x.sort(key=func)
                return (sorted(x[:], key=func), new_x)

            @compiler
            def foo(x):

                def closure_escapee_key(z):
                    return 1.0 / z
                return bar(x, closure_escapee_key)
            return foo
        self.assertPreciseEqual(gen(njit)(a[:]), gen(nop_compiler)(a[:]))

    def test_04(self):
        a = ['a', 'b', 'B', 'b', 'C', 'A']

        @njit
        def external_key(z):
            return z.upper()

        @njit
        def foo(x, key=None):
            new_x = x[:]
            new_x.sort(key=key)
            return (sorted(x[:], key=key), new_x)
        self.assertPreciseEqual(foo(a[:]), foo.py_func(a[:]))
        self.assertPreciseEqual(foo(a[:], external_key), foo.py_func(a[:], external_key))

    def test_05(self):
        a = ['a', 'b', 'B', 'b', 'C', 'A']

        @njit
        def external_key(z):
            return z.upper()

        @njit
        def foo(x, key=None, reverse=False):
            new_x = x[:]
            new_x.sort(key=key, reverse=reverse)
            return (sorted(x[:], key=key, reverse=reverse), new_x)
        for key, rev in itertools.product((None, external_key), (True, False, 1, -12, 0)):
            self.assertPreciseEqual(foo(a[:], key, rev), foo.py_func(a[:], key, rev))

    def test_optional_on_key(self):
        a = [3, 1, 4, 1, 5, 9]

        @njit
        def foo(x, predicate):
            if predicate:

                def closure_key(z):
                    return 1.0 / z
            else:
                closure_key = None
            new_x = x[:]
            new_x.sort(key=closure_key)
            return (sorted(x[:], key=closure_key), new_x)
        with self.assertRaises(errors.TypingError) as raises:
            TF = True
            foo(a[:], TF)
        msg = 'Key must concretely be None or a Numba JIT compiled function'
        self.assertIn(msg, str(raises.exception))

    def test_exceptions_sorted(self):

        @njit
        def foo_sorted(x, key=None, reverse=False):
            return sorted(x[:], key=key, reverse=reverse)

        @njit
        def foo_sort(x, key=None, reverse=False):
            new_x = x[:]
            new_x.sort(key=key, reverse=reverse)
            return new_x

        @njit
        def external_key(z):
            return 1.0 / z
        a = [3, 1, 4, 1, 5, 9]
        for impl in (foo_sort, foo_sorted):
            with self.assertRaises(errors.TypingError) as raises:
                impl(a, key='illegal')
            expect = 'Key must be None or a Numba JIT compiled function'
            self.assertIn(expect, str(raises.exception))
            with self.assertRaises(errors.TypingError) as raises:
                impl(a, key=external_key, reverse='go backwards')
            expect = "an integer is required for 'reverse'"
            self.assertIn(expect, str(raises.exception))