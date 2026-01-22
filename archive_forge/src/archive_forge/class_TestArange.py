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
class TestArange:

    def test_infinite(self):
        assert_raises_regex(ValueError, 'size exceeded', np.arange, 0, np.inf)

    def test_nan_step(self):
        assert_raises_regex(ValueError, 'cannot compute length', np.arange, 0, 1, np.nan)

    def test_zero_step(self):
        assert_raises(ZeroDivisionError, np.arange, 0, 10, 0)
        assert_raises(ZeroDivisionError, np.arange, 0.0, 10.0, 0.0)
        assert_raises(ZeroDivisionError, np.arange, 0, 0, 0)
        assert_raises(ZeroDivisionError, np.arange, 0.0, 0.0, 0.0)

    def test_require_range(self):
        assert_raises(TypeError, np.arange)
        assert_raises(TypeError, np.arange, step=3)
        assert_raises(TypeError, np.arange, dtype='int64')
        assert_raises(TypeError, np.arange, start=4)

    def test_start_stop_kwarg(self):
        keyword_stop = np.arange(stop=3)
        keyword_zerotostop = np.arange(start=0, stop=3)
        keyword_start_stop = np.arange(start=3, stop=9)
        assert len(keyword_stop) == 3
        assert len(keyword_zerotostop) == 3
        assert len(keyword_start_stop) == 6
        assert_array_equal(keyword_stop, keyword_zerotostop)

    def test_arange_booleans(self):
        res = np.arange(False, dtype=bool)
        assert_array_equal(res, np.array([], dtype='bool'))
        res = np.arange(True, dtype='bool')
        assert_array_equal(res, [False])
        res = np.arange(2, dtype='bool')
        assert_array_equal(res, [False, True])
        res = np.arange(6, 8, dtype='bool')
        assert_array_equal(res, [True, True])
        with pytest.raises(TypeError):
            np.arange(3, dtype='bool')

    @pytest.mark.parametrize('dtype', ['S3', 'U', '5i'])
    def test_rejects_bad_dtypes(self, dtype):
        dtype = np.dtype(dtype)
        DType_name = re.escape(str(type(dtype)))
        with pytest.raises(TypeError, match=f'arange\\(\\) not supported for inputs .* {DType_name}'):
            np.arange(2, dtype=dtype)

    def test_rejects_strings(self):
        DType_name = re.escape(str(type(np.array('a').dtype)))
        with pytest.raises(TypeError, match=f'arange\\(\\) not supported for inputs .* {DType_name}'):
            np.arange('a', 'b')

    def test_byteswapped(self):
        res_be = np.arange(1, 1000, dtype='>i4')
        res_le = np.arange(1, 1000, dtype='<i4')
        assert res_be.dtype == '>i4'
        assert res_le.dtype == '<i4'
        assert_array_equal(res_le, res_be)

    @pytest.mark.parametrize('which', [0, 1, 2])
    def test_error_paths_and_promotion(self, which):
        args = [0, 1, 2]
        args[which] = np.float64(2.0)
        assert np.arange(*args).dtype == np.float64
        args[which] = [None, []]
        with pytest.raises(ValueError):
            np.arange(*args)