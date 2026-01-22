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
class TestArgmaxArgminCommon:
    sizes = [(), (3,), (3, 2), (2, 3), (3, 3), (2, 3, 4), (4, 3, 2), (1, 2, 3, 4), (2, 3, 4, 1), (3, 4, 1, 2), (4, 1, 2, 3), (64,), (128,), (256,)]

    @pytest.mark.parametrize('size, axis', itertools.chain(*[[(size, axis) for axis in list(range(-len(size), len(size))) + [None]] for size in sizes]))
    @pytest.mark.parametrize('method', [np.argmax, np.argmin])
    def test_np_argmin_argmax_keepdims(self, size, axis, method):
        arr = np.random.normal(size=size)
        if axis is None:
            new_shape = [1 for _ in range(len(size))]
        else:
            new_shape = list(size)
            new_shape[axis] = 1
        new_shape = tuple(new_shape)
        _res_orig = method(arr, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        res = method(arr, axis=axis, keepdims=True)
        assert_equal(res, res_orig)
        assert_(res.shape == new_shape)
        outarray = np.empty(res.shape, dtype=res.dtype)
        res1 = method(arr, axis=axis, out=outarray, keepdims=True)
        assert_(res1 is outarray)
        assert_equal(res, outarray)
        if len(size) > 0:
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)
        if axis is None:
            new_shape = [1 for _ in range(len(size))]
        else:
            new_shape = list(size)[::-1]
            new_shape[axis] = 1
        new_shape = tuple(new_shape)
        _res_orig = method(arr.T, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        res = method(arr.T, axis=axis, keepdims=True)
        assert_equal(res, res_orig)
        assert_(res.shape == new_shape)
        outarray = np.empty(new_shape[::-1], dtype=res.dtype)
        outarray = outarray.T
        res1 = method(arr.T, axis=axis, out=outarray, keepdims=True)
        assert_(res1 is outarray)
        assert_equal(res, outarray)
        if len(size) > 0:
            with pytest.raises(ValueError):
                method(arr[0], axis=axis, out=outarray, keepdims=True)
        if len(size) > 0:
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)

    @pytest.mark.parametrize('method', ['max', 'min'])
    def test_all(self, method):
        a = np.random.normal(0, 1, (4, 5, 6, 7, 8))
        arg_method = getattr(a, 'arg' + method)
        val_method = getattr(a, method)
        for i in range(a.ndim):
            a_maxmin = val_method(i)
            aarg_maxmin = arg_method(i)
            axes = list(range(a.ndim))
            axes.remove(i)
            assert_(np.all(a_maxmin == aarg_maxmin.choose(*a.transpose(i, *axes))))

    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    def test_output_shape(self, method):
        a = np.ones((10, 5))
        arg_method = getattr(a, method)
        out = np.ones(11, dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)
        out = np.ones((2, 5), dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)
        out = np.ones((1, 10), dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)
        out = np.ones(10, dtype=np.int_)
        arg_method(-1, out=out)
        assert_equal(out, arg_method(-1))

    @pytest.mark.parametrize('ndim', [0, 1])
    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    def test_ret_is_out(self, ndim, method):
        a = np.ones((4,) + (256,) * ndim)
        arg_method = getattr(a, method)
        out = np.empty((256,) * ndim, dtype=np.intp)
        ret = arg_method(axis=0, out=out)
        assert ret is out

    @pytest.mark.parametrize('np_array, method, idx, val', [(np.zeros, 'argmax', 5942, 'as'), (np.ones, 'argmin', 6001, '0')])
    def test_unicode(self, np_array, method, idx, val):
        d = np_array(6031, dtype='<U9')
        arg_method = getattr(d, method)
        d[idx] = val
        assert_equal(arg_method(), idx)

    @pytest.mark.parametrize('arr_method, np_method', [('argmax', np.argmax), ('argmin', np.argmin)])
    def test_np_vs_ndarray(self, arr_method, np_method):
        a = np.random.normal(size=(2, 3))
        arg_method = getattr(a, arr_method)
        out1 = np.zeros(2, dtype=int)
        out2 = np.zeros(2, dtype=int)
        assert_equal(arg_method(1, out1), np_method(a, 1, out2))
        assert_equal(out1, out2)
        out1 = np.zeros(3, dtype=int)
        out2 = np.zeros(3, dtype=int)
        assert_equal(arg_method(out=out1, axis=0), np_method(a, out=out2, axis=0))
        assert_equal(out1, out2)

    @pytest.mark.leaks_references(reason='replaces None with NULL.')
    @pytest.mark.parametrize('method, vals', [('argmax', (10, 30)), ('argmin', (30, 10))])
    def test_object_with_NULLs(self, method, vals):
        a = np.empty(4, dtype='O')
        arg_method = getattr(a, method)
        ctypes.memset(a.ctypes.data, 0, a.nbytes)
        assert_equal(arg_method(), 0)
        a[3] = vals[0]
        assert_equal(arg_method(), 3)
        a[1] = vals[1]
        assert_equal(arg_method(), 1)