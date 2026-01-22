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
class TestUnicodeEncoding:
    """
    Tests for encoding related bugs, such as UCS2 vs UCS4, round-tripping
    issues, etc
    """

    def test_round_trip(self):
        """ Tests that GETITEM, SETITEM, and PyArray_Scalar roundtrip """
        arr = np.zeros(shape=(), dtype='U1')
        for i in range(1, sys.maxunicode + 1):
            expected = chr(i)
            arr[()] = expected
            assert arr[()] == expected
            assert arr.item() == expected

    def test_assign_scalar(self):
        l = np.array(['aa', 'bb'])
        l[:] = np.str_('cc')
        assert_equal(l, ['cc', 'cc'])

    def test_fill_scalar(self):
        l = np.array(['aa', 'bb'])
        l.fill(np.str_('cc'))
        assert_equal(l, ['cc', 'cc'])