import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
class TestOut:

    def test_out_subok(self):
        for subok in (True, False):
            a = np.array(0.5)
            o = np.empty(())
            r = np.add(a, 2, o, subok=subok)
            assert_(r is o)
            r = np.add(a, 2, out=o, subok=subok)
            assert_(r is o)
            r = np.add(a, 2, out=(o,), subok=subok)
            assert_(r is o)
            d = np.array(5.7)
            o1 = np.empty(())
            o2 = np.empty((), dtype=np.int32)
            r1, r2 = np.frexp(d, o1, None, subok=subok)
            assert_(r1 is o1)
            r1, r2 = np.frexp(d, None, o2, subok=subok)
            assert_(r2 is o2)
            r1, r2 = np.frexp(d, o1, o2, subok=subok)
            assert_(r1 is o1)
            assert_(r2 is o2)
            r1, r2 = np.frexp(d, out=(o1, None), subok=subok)
            assert_(r1 is o1)
            r1, r2 = np.frexp(d, out=(None, o2), subok=subok)
            assert_(r2 is o2)
            r1, r2 = np.frexp(d, out=(o1, o2), subok=subok)
            assert_(r1 is o1)
            assert_(r2 is o2)
            with assert_raises(TypeError):
                r1, r2 = np.frexp(d, out=o1, subok=subok)
            assert_raises(TypeError, np.add, a, 2, o, o, subok=subok)
            assert_raises(TypeError, np.add, a, 2, o, out=o, subok=subok)
            assert_raises(TypeError, np.add, a, 2, None, out=o, subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=(o, o), subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=(), subok=subok)
            assert_raises(TypeError, np.add, a, 2, [], subok=subok)
            assert_raises(TypeError, np.add, a, 2, out=[], subok=subok)
            assert_raises(TypeError, np.add, a, 2, out=([],), subok=subok)
            o.flags.writeable = False
            assert_raises(ValueError, np.add, a, 2, o, subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=o, subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=(o,), subok=subok)

    def test_out_wrap_subok(self):

        class ArrayWrap(np.ndarray):
            __array_priority__ = 10

            def __new__(cls, arr):
                return np.asarray(arr).view(cls).copy()

            def __array_wrap__(self, arr, context):
                return arr.view(type(self))
        for subok in (True, False):
            a = ArrayWrap([0.5])
            r = np.add(a, 2, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)
            r = np.add(a, 2, None, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)
            r = np.add(a, 2, out=None, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)
            r = np.add(a, 2, out=(None,), subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)
            d = ArrayWrap([5.7])
            o1 = np.empty((1,))
            o2 = np.empty((1,), dtype=np.int32)
            r1, r2 = np.frexp(d, o1, subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)
            r1, r2 = np.frexp(d, o1, None, subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)
            r1, r2 = np.frexp(d, None, o2, subok=subok)
            if subok:
                assert_(isinstance(r1, ArrayWrap))
            else:
                assert_(type(r1) == np.ndarray)
            r1, r2 = np.frexp(d, out=(o1, None), subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)
            r1, r2 = np.frexp(d, out=(None, o2), subok=subok)
            if subok:
                assert_(isinstance(r1, ArrayWrap))
            else:
                assert_(type(r1) == np.ndarray)
            with assert_raises(TypeError):
                r1, r2 = np.frexp(d, out=o1, subok=subok)