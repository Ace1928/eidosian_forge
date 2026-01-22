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
class TestWritebackIfCopy:

    def test_argmax_with_out(self):
        mat = np.eye(5)
        out = np.empty(5, dtype='i2')
        res = np.argmax(mat, 0, out=out)
        assert_equal(res, range(5))

    def test_argmin_with_out(self):
        mat = -np.eye(5)
        out = np.empty(5, dtype='i2')
        res = np.argmin(mat, 0, out=out)
        assert_equal(res, range(5))

    def test_insert_noncontiguous(self):
        a = np.arange(6).reshape(2, 3).T
        np.place(a, a > 2, [44, 55])
        assert_equal(a, np.array([[0, 44], [1, 55], [2, 44]]))
        assert_raises(ValueError, np.place, a, a > 20, [])

    def test_put_noncontiguous(self):
        a = np.arange(6).reshape(2, 3).T
        np.put(a, [0, 2], [44, 55])
        assert_equal(a, np.array([[44, 3], [55, 4], [2, 5]]))

    def test_putmask_noncontiguous(self):
        a = np.arange(6).reshape(2, 3).T
        np.putmask(a, a > 2, a ** 2)
        assert_equal(a, np.array([[0, 9], [1, 16], [2, 25]]))

    def test_take_mode_raise(self):
        a = np.arange(6, dtype='int')
        out = np.empty(2, dtype='int')
        np.take(a, [0, 2], out=out, mode='raise')
        assert_equal(out, np.array([0, 2]))

    def test_choose_mod_raise(self):
        a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        out = np.empty((3, 3), dtype='int')
        choices = [-10, 10]
        np.choose(a, choices, out=out, mode='raise')
        assert_equal(out, np.array([[10, -10, 10], [-10, 10, -10], [10, -10, 10]]))

    def test_flatiter__array__(self):
        a = np.arange(9).reshape(3, 3)
        b = a.T.flat
        c = b.__array__()
        del c

    def test_dot_out(self):
        a = np.arange(9, dtype=float).reshape(3, 3)
        b = np.dot(a, a, out=a)
        assert_equal(b, np.array([[15, 18, 21], [42, 54, 66], [69, 90, 111]]))

    def test_view_assign(self):
        from numpy.core._multiarray_tests import npy_create_writebackifcopy, npy_resolve
        arr = np.arange(9).reshape(3, 3).T
        arr_wb = npy_create_writebackifcopy(arr)
        assert_(arr_wb.flags.writebackifcopy)
        assert_(arr_wb.base is arr)
        arr_wb[...] = -100
        npy_resolve(arr_wb)
        assert_equal(arr, -100)
        assert_(arr_wb.ctypes.data != 0)
        assert_equal(arr_wb.base, None)
        arr_wb[...] = 100
        assert_equal(arr, -100)

    @pytest.mark.leaks_references(reason='increments self in dealloc; ignore since deprecated path.')
    def test_dealloc_warning(self):
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            arr = np.arange(9).reshape(3, 3)
            v = arr.T
            _multiarray_tests.npy_abuse_writebackifcopy(v)
            assert len(sup.log) == 1

    def test_view_discard_refcount(self):
        from numpy.core._multiarray_tests import npy_create_writebackifcopy, npy_discard
        arr = np.arange(9).reshape(3, 3).T
        orig = arr.copy()
        if HAS_REFCOUNT:
            arr_cnt = sys.getrefcount(arr)
        arr_wb = npy_create_writebackifcopy(arr)
        assert_(arr_wb.flags.writebackifcopy)
        assert_(arr_wb.base is arr)
        arr_wb[...] = -100
        npy_discard(arr_wb)
        assert_equal(arr, orig)
        assert_(arr_wb.ctypes.data != 0)
        assert_equal(arr_wb.base, None)
        if HAS_REFCOUNT:
            assert_equal(arr_cnt, sys.getrefcount(arr))
        arr_wb[...] = 100
        assert_equal(arr, orig)