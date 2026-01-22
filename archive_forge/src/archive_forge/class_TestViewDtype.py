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
class TestViewDtype:
    """
    Verify that making a view of a non-contiguous array works as expected.
    """

    def test_smaller_dtype_multiple(self):
        x = np.arange(10, dtype='<i4')[::2]
        with pytest.raises(ValueError, match='the last axis must be contiguous'):
            x.view('<i2')
        expected = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0]]
        assert_array_equal(x[:, np.newaxis].view('<i2'), expected)

    def test_smaller_dtype_not_multiple(self):
        x = np.arange(5, dtype='<i4')[::2]
        with pytest.raises(ValueError, match='the last axis must be contiguous'):
            x.view('S3')
        with pytest.raises(ValueError, match='When changing to a smaller dtype'):
            x[:, np.newaxis].view('S3')
        expected = [[b''], [b'\x02'], [b'\x04']]
        assert_array_equal(x[:, np.newaxis].view('S4'), expected)

    def test_larger_dtype_multiple(self):
        x = np.arange(20, dtype='<i2').reshape(10, 2)[::2, :]
        expected = np.array([[65536], [327684], [589832], [851980], [1114128]], dtype='<i4')
        assert_array_equal(x.view('<i4'), expected)

    def test_larger_dtype_not_multiple(self):
        x = np.arange(20, dtype='<i2').reshape(10, 2)[::2, :]
        with pytest.raises(ValueError, match='When changing to a larger dtype'):
            x.view('S3')
        expected = [[b'\x00\x00\x01'], [b'\x04\x00\x05'], [b'\x08\x00\t'], [b'\x0c\x00\r'], [b'\x10\x00\x11']]
        assert_array_equal(x.view('S4'), expected)

    def test_f_contiguous(self):
        x = np.arange(4 * 3, dtype='<i4').reshape(4, 3).T
        with pytest.raises(ValueError, match='the last axis must be contiguous'):
            x.view('<i2')

    def test_non_c_contiguous(self):
        x = np.arange(2 * 3 * 4, dtype='i1').reshape(2, 3, 4).transpose(1, 0, 2)
        expected = [[[256, 770], [3340, 3854]], [[1284, 1798], [4368, 4882]], [[2312, 2826], [5396, 5910]]]
        assert_array_equal(x.view('<i2'), expected)