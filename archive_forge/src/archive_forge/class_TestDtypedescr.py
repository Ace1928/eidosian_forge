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
class TestDtypedescr:

    def test_construction(self):
        d1 = np.dtype('i4')
        assert_equal(d1, np.dtype(np.int32))
        d2 = np.dtype('f8')
        assert_equal(d2, np.dtype(np.float64))

    def test_byteorders(self):
        assert_(np.dtype('<i4') != np.dtype('>i4'))
        assert_(np.dtype([('a', '<i4')]) != np.dtype([('a', '>i4')]))

    def test_structured_non_void(self):
        fields = [('a', '<i2'), ('b', '<i2')]
        dt_int = np.dtype(('i4', fields))
        assert_equal(str(dt_int), "(numpy.int32, [('a', '<i2'), ('b', '<i2')])")
        arr_int = np.zeros(4, dt_int)
        assert_equal(repr(arr_int), "array([0, 0, 0, 0], dtype=(numpy.int32, [('a', '<i2'), ('b', '<i2')]))")