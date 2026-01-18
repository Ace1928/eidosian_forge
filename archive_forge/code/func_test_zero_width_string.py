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
def test_zero_width_string(self):
    dt = np.dtype([('I', int), ('S', 'S0')])
    x = np.zeros(4, dtype=dt)
    assert_equal(x['S'], [b'', b'', b'', b''])
    assert_equal(x['S'].itemsize, 0)
    x['S'] = ['a', 'b', 'c', 'd']
    assert_equal(x['S'], [b'', b'', b'', b''])
    assert_equal(x['I'], [0, 0, 0, 0])
    x['S'][x['I'] == 0] = 'hello'
    assert_equal(x['S'], [b'', b'', b'', b''])
    assert_equal(x['I'], [0, 0, 0, 0])
    x['S'] = 'A'
    assert_equal(x['S'], [b'', b'', b'', b''])
    assert_equal(x['I'], [0, 0, 0, 0])
    y = np.ndarray(4, dtype=x['S'].dtype)
    assert_equal(y.itemsize, 0)
    assert_equal(x['S'], y)
    assert_equal(np.zeros(4, dtype=[('a', 'S0,S0'), ('b', 'u1')])['a'].itemsize, 0)
    assert_equal(np.empty(3, dtype='S0,S0').itemsize, 0)
    assert_equal(np.zeros(4, dtype='S0,u1')['f0'].itemsize, 0)
    xx = x['S'].reshape((2, 2))
    assert_equal(xx.itemsize, 0)
    assert_equal(xx, [[b'', b''], [b'', b'']])
    assert_equal(xx[:].dtype, xx.dtype)
    assert_array_equal(eval(repr(xx), dict(array=np.array)), xx)
    b = io.BytesIO()
    np.save(b, xx)
    b.seek(0)
    yy = np.load(b)
    assert_equal(yy.itemsize, 0)
    assert_equal(xx, yy)
    with temppath(suffix='.npy') as tmp:
        np.save(tmp, xx)
        yy = np.load(tmp)
        assert_equal(yy.itemsize, 0)
        assert_equal(xx, yy)