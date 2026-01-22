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
class TestCTypes:

    def test_ctypes_is_available(self):
        test_arr = np.array([[1, 2, 3], [4, 5, 6]])
        assert_equal(ctypes, test_arr.ctypes._ctypes)
        assert_equal(tuple(test_arr.ctypes.shape), (2, 3))

    def test_ctypes_is_not_available(self):
        from numpy.core import _internal
        _internal.ctypes = None
        try:
            test_arr = np.array([[1, 2, 3], [4, 5, 6]])
            assert_(isinstance(test_arr.ctypes._ctypes, _internal._missing_ctypes))
            assert_equal(tuple(test_arr.ctypes.shape), (2, 3))
        finally:
            _internal.ctypes = ctypes

    def _make_readonly(x):
        x.flags.writeable = False
        return x

    @pytest.mark.parametrize('arr', [np.array([1, 2, 3]), np.array([['one', 'two'], ['three', 'four']]), np.array((1, 2), dtype='i4,i4'), np.zeros((2,), dtype=np.dtype(dict(formats=['<i4', '<i4'], names=['a', 'b'], offsets=[0, 2], itemsize=6))), np.array([None], dtype=object), np.array([]), np.empty((0, 0)), _make_readonly(np.array([1, 2, 3]))], ids=['1d', '2d', 'structured', 'overlapping', 'object', 'empty', 'empty-2d', 'readonly'])
    def test_ctypes_data_as_holds_reference(self, arr):
        arr = arr.copy()
        arr_ref = weakref.ref(arr)
        ctypes_ptr = arr.ctypes.data_as(ctypes.c_void_p)
        del arr
        break_cycles()
        assert_(arr_ref() is not None, 'ctypes pointer did not hold onto a reference')
        del ctypes_ptr
        if IS_PYPY:
            break_cycles()
        assert_(arr_ref() is None, 'unknowable whether ctypes pointer holds a reference')

    def test_ctypes_as_parameter_holds_reference(self):
        arr = np.array([None]).copy()
        arr_ref = weakref.ref(arr)
        ctypes_ptr = arr.ctypes._as_parameter_
        del arr
        break_cycles()
        assert_(arr_ref() is not None, 'ctypes pointer did not hold onto a reference')
        del ctypes_ptr
        if IS_PYPY:
            break_cycles()
        assert_(arr_ref() is None, 'unknowable whether ctypes pointer holds a reference')