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
def test_structured_cast_promotion_fieldorder(self):
    A = ('a', '<i8')
    B = ('b', '>i8')
    ab = np.array([(1, 2)], dtype=[A, B])
    ba = np.array([(1, 2)], dtype=[B, A])
    assert_raises(TypeError, np.concatenate, ab, ba)
    assert_raises(TypeError, np.result_type, ab.dtype, ba.dtype)
    assert_raises(TypeError, np.promote_types, ab.dtype, ba.dtype)
    assert_equal(np.promote_types(ab.dtype, ba[['a', 'b']].dtype), repack_fields(ab.dtype.newbyteorder('N')))
    assert_equal(np.can_cast(ab.dtype, ba.dtype), True)
    assert_equal(ab.astype(ba.dtype).dtype, ba.dtype)
    assert_equal(np.can_cast('f8,i8', [('f0', 'f8'), ('f1', 'i8')]), True)
    assert_equal(np.can_cast('f8,i8', [('f1', 'f8'), ('f0', 'i8')]), True)
    assert_equal(np.can_cast('f8,i8', [('f1', 'i8'), ('f0', 'f8')]), False)
    assert_equal(np.can_cast('f8,i8', [('f1', 'i8'), ('f0', 'f8')], casting='unsafe'), True)
    ab[:] = ba
    dt1 = np.dtype([('', 'i4')])
    dt2 = np.dtype([('', 'i8')])
    assert_equal(np.promote_types(dt1, dt2), np.dtype([('f0', 'i8')]))
    assert_equal(np.promote_types(dt2, dt1), np.dtype([('f0', 'i8')]))
    assert_raises(TypeError, np.promote_types, dt1, np.dtype([('', 'V3')]))
    assert_equal(np.promote_types('i4,f8', 'i8,f4'), np.dtype([('f0', 'i8'), ('f1', 'f8')]))
    dt1nest = np.dtype([('', dt1)])
    dt2nest = np.dtype([('', dt2)])
    assert_equal(np.promote_types(dt1nest, dt2nest), np.dtype([('f0', np.dtype([('f0', 'i8')]))]))
    dt = np.dtype({'names': ['x'], 'formats': ['i4'], 'offsets': [8]})
    a = np.ones(3, dtype=dt)
    assert_equal(np.concatenate([a, a]).dtype, np.dtype([('x', 'i4')]))