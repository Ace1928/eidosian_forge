from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
class TestWhere:

    def test_basic(self):
        dts = [bool, np.int16, np.int32, np.int64, np.double, np.complex128, np.longdouble, np.clongdouble]
        for dt in dts:
            c = np.ones(53, dtype=bool)
            assert_equal(np.where(c, dt(0), dt(1)), dt(0))
            assert_equal(np.where(~c, dt(0), dt(1)), dt(1))
            assert_equal(np.where(True, dt(0), dt(1)), dt(0))
            assert_equal(np.where(False, dt(0), dt(1)), dt(1))
            d = np.ones_like(c).astype(dt)
            e = np.zeros_like(d)
            r = d.astype(dt)
            c[7] = False
            r[7] = e[7]
            assert_equal(np.where(c, e, e), e)
            assert_equal(np.where(c, d, e), r)
            assert_equal(np.where(c, d, e[0]), r)
            assert_equal(np.where(c, d[0], e), r)
            assert_equal(np.where(c[::2], d[::2], e[::2]), r[::2])
            assert_equal(np.where(c[1::2], d[1::2], e[1::2]), r[1::2])
            assert_equal(np.where(c[::3], d[::3], e[::3]), r[::3])
            assert_equal(np.where(c[1::3], d[1::3], e[1::3]), r[1::3])
            assert_equal(np.where(c[::-2], d[::-2], e[::-2]), r[::-2])
            assert_equal(np.where(c[::-3], d[::-3], e[::-3]), r[::-3])
            assert_equal(np.where(c[1::-3], d[1::-3], e[1::-3]), r[1::-3])

    def test_exotic(self):
        assert_array_equal(np.where(True, None, None), np.array(None))
        m = np.array([], dtype=bool).reshape(0, 3)
        b = np.array([], dtype=np.float64).reshape(0, 3)
        assert_array_equal(np.where(m, 0, b), np.array([]).reshape(0, 3))
        d = np.array([-1.34, -0.16, -0.54, -0.31, -0.08, -0.95, 0.0, 0.313, 0.547, -0.18, 0.876, 0.236, 1.969, 0.31, 0.699, 1.013, 1.267, 0.229, -1.39, 0.487])
        nan = float('NaN')
        e = np.array(['5z', '0l', nan, 'Wz', nan, nan, 'Xq', 'cs', nan, nan, 'QN', nan, nan, 'Fd', nan, nan, 'kp', nan, '36', 'i1'], dtype=object)
        m = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
        r = e[:]
        r[np.where(m)] = d[np.where(m)]
        assert_array_equal(np.where(m, d, e), r)
        r = e[:]
        r[np.where(~m)] = d[np.where(~m)]
        assert_array_equal(np.where(m, e, d), r)
        assert_array_equal(np.where(m, e, e), e)
        d = np.array([1.0, 2.0], dtype=np.float32)
        e = float('NaN')
        assert_equal(np.where(True, d, e).dtype, np.float32)
        e = float('Infinity')
        assert_equal(np.where(True, d, e).dtype, np.float32)
        e = float('-Infinity')
        assert_equal(np.where(True, d, e).dtype, np.float32)
        e = float(1e+150)
        assert_equal(np.where(True, d, e).dtype, np.float64)

    def test_ndim(self):
        c = [True, False]
        a = np.zeros((2, 25))
        b = np.ones((2, 25))
        r = np.where(np.array(c)[:, np.newaxis], a, b)
        assert_array_equal(r[0], a[0])
        assert_array_equal(r[1], b[0])
        a = a.T
        b = b.T
        r = np.where(c, a, b)
        assert_array_equal(r[:, 0], a[:, 0])
        assert_array_equal(r[:, 1], b[:, 0])

    def test_dtype_mix(self):
        c = np.array([False, True, False, False, False, False, True, False, False, False, True, False])
        a = np.uint32(1)
        b = np.array([5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0], dtype=np.float64)
        r = np.array([5.0, 1.0, 3.0, 2.0, -1.0, -4.0, 1.0, -10.0, 10.0, 1.0, 1.0, 3.0], dtype=np.float64)
        assert_equal(np.where(c, a, b), r)
        a = a.astype(np.float32)
        b = b.astype(np.int64)
        assert_equal(np.where(c, a, b), r)
        c = c.astype(int)
        c[c != 0] = 34242324
        assert_equal(np.where(c, a, b), r)
        tmpmask = c != 0
        c[c == 0] = 41247212
        c[tmpmask] = 0
        assert_equal(np.where(c, b, a), r)

    def test_foreign(self):
        c = np.array([False, True, False, False, False, False, True, False, False, False, True, False])
        r = np.array([5.0, 1.0, 3.0, 2.0, -1.0, -4.0, 1.0, -10.0, 10.0, 1.0, 1.0, 3.0], dtype=np.float64)
        a = np.ones(1, dtype='>i4')
        b = np.array([5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0], dtype=np.float64)
        assert_equal(np.where(c, a, b), r)
        b = b.astype('>f8')
        assert_equal(np.where(c, a, b), r)
        a = a.astype('<i4')
        assert_equal(np.where(c, a, b), r)
        c = c.astype('>i4')
        assert_equal(np.where(c, a, b), r)

    def test_error(self):
        c = [True, True]
        a = np.ones((4, 5))
        b = np.ones((5, 5))
        assert_raises(ValueError, np.where, c, a, a)
        assert_raises(ValueError, np.where, c[0], a, b)

    def test_string(self):
        a = np.array('abc')
        b = np.array('x' * 753)
        assert_equal(np.where(True, a, b), 'abc')
        assert_equal(np.where(False, b, a), 'abc')
        a = np.array('abcd')
        b = np.array('x' * 8)
        assert_equal(np.where(True, a, b), 'abcd')
        assert_equal(np.where(False, b, a), 'abcd')

    def test_empty_result(self):
        x = np.zeros((1, 1))
        ibad = np.vstack(np.where(x == 99.0))
        assert_array_equal(ibad, np.atleast_2d(np.array([[], []], dtype=np.intp)))

    def test_largedim(self):
        shape = [10, 2, 3, 4, 5, 6]
        np.random.seed(2)
        array = np.random.rand(*shape)
        for i in range(10):
            benchmark = array.nonzero()
            result = array.nonzero()
            assert_array_equal(benchmark, result)

    def test_kwargs(self):
        a = np.zeros(1)
        with assert_raises(TypeError):
            np.where(a, x=a, y=a)