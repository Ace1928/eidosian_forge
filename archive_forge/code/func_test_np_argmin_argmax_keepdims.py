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