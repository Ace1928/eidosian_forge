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
class TestVdot:

    def test_basic(self):
        dt_numeric = np.typecodes['AllFloat'] + np.typecodes['AllInteger']
        dt_complex = np.typecodes['Complex']
        a = np.eye(3)
        for dt in dt_numeric + 'O':
            b = a.astype(dt)
            res = np.vdot(b, b)
            assert_(np.isscalar(res))
            assert_equal(np.vdot(b, b), 3)
        a = np.eye(3) * 1j
        for dt in dt_complex + 'O':
            b = a.astype(dt)
            res = np.vdot(b, b)
            assert_(np.isscalar(res))
            assert_equal(np.vdot(b, b), 3)
        b = np.eye(3, dtype=bool)
        res = np.vdot(b, b)
        assert_(np.isscalar(res))
        assert_equal(np.vdot(b, b), True)

    def test_vdot_array_order(self):
        a = np.array([[1, 2], [3, 4]], order='C')
        b = np.array([[1, 2], [3, 4]], order='F')
        res = np.vdot(a, a)
        assert_equal(np.vdot(a, b), res)
        assert_equal(np.vdot(b, a), res)
        assert_equal(np.vdot(b, b), res)

    def test_vdot_uncontiguous(self):
        for size in [2, 1000]:
            a = np.zeros((size, 2, 2))
            b = np.zeros((size, 2, 2))
            a[:, 0, 0] = np.arange(size)
            b[:, 0, 0] = np.arange(size) + 1
            a = a[..., 0]
            b = b[..., 0]
            assert_equal(np.vdot(a, b), np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a, b.copy()), np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a.copy(), b), np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a.copy('F'), b), np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a, b.copy('F')), np.vdot(a.flatten(), b.flatten()))