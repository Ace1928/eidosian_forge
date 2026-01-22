from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
class TestListReflection(MemoryLeakMixin, TestCase):
    """
    Test reflection of native Numba lists on Python list objects.
    """

    def check_reflection(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        samples = [([1.0, 2.0, 3.0, 4.0], [0.0]), ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0])]
        for dest, src in samples:
            expected = list(dest)
            got = list(dest)
            pyres = pyfunc(expected, src)
            with self.assertRefCount(got, src):
                cres = cfunc(got, src)
                self.assertPreciseEqual(cres, pyres)
                self.assertPreciseEqual(expected, got)
                self.assertEqual(pyres[0] is expected, cres[0] is got)
                del pyres, cres

    def test_reflect_simple(self):
        self.check_reflection(reflect_simple)

    def test_reflect_conditional(self):
        self.check_reflection(reflect_conditional)

    def test_reflect_exception(self):
        """
        When the function exits with an exception, lists should still be
        reflected.
        """
        pyfunc = reflect_exception
        cfunc = jit(nopython=True)(pyfunc)
        l = [1, 2, 3]
        with self.assertRefCount(l):
            with self.assertRaises(ZeroDivisionError):
                cfunc(l)
            self.assertPreciseEqual(l, [1, 2, 3, 42])

    def test_reflect_same_list(self):
        """
        When the same list object is reflected twice, behaviour should
        be consistent.
        """
        pyfunc = reflect_dual
        cfunc = jit(nopython=True)(pyfunc)
        pylist = [1, 2, 3]
        clist = pylist[:]
        expected = pyfunc(pylist, pylist)
        got = cfunc(clist, clist)
        self.assertPreciseEqual(expected, got)
        self.assertPreciseEqual(pylist, clist)
        self.assertRefCountEqual(pylist, clist)

    def test_reflect_clean(self):
        """
        When the list wasn't mutated, no reflection should take place.
        """
        cfunc = jit(nopython=True)(noop)
        l = [12.5j]
        ids = [id(x) for x in l]
        cfunc(l)
        self.assertEqual([id(x) for x in l], ids)