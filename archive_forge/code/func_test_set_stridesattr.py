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
def test_set_stridesattr(self):
    x = self.one

    def make_array(size, offset, strides):
        try:
            r = np.ndarray([size], dtype=int, buffer=x, offset=offset * x.itemsize)
        except Exception as e:
            raise RuntimeError(e)
        r.strides = strides = strides * x.itemsize
        return r
    assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
    assert_equal(make_array(7, 3, 1), np.array([3, 4, 5, 6, 7, 8, 9]))
    assert_raises(ValueError, make_array, 4, 4, -2)
    assert_raises(ValueError, make_array, 4, 2, -1)
    assert_raises(RuntimeError, make_array, 8, 3, 1)
    x = np.lib.stride_tricks.as_strided(np.arange(1), (10, 10), (0, 0))

    def set_strides(arr, strides):
        arr.strides = strides
    assert_raises(ValueError, set_strides, x, (10 * x.itemsize, x.itemsize))
    x = np.lib.stride_tricks.as_strided(np.arange(10, dtype=np.int8)[-1], shape=(10,), strides=(-1,))
    assert_raises(ValueError, set_strides, x[::-1], -1)
    a = x[::-1]
    a.strides = 1
    a[::2].strides = 2
    arr_0d = np.array(0)
    arr_0d.strides = ()
    assert_raises(TypeError, set_strides, arr_0d, None)