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
def test_searchsorted_type_specific(self):
    types = ''.join((np.typecodes['AllInteger'], np.typecodes['AllFloat'], np.typecodes['Datetime'], '?O'))
    for dt in types:
        if dt == 'M':
            dt = 'M8[D]'
        if dt == '?':
            a = np.arange(2, dtype=dt)
            out = np.arange(2)
        else:
            a = np.arange(0, 5, dtype=dt)
            out = np.arange(5)
        b = a.searchsorted(a, 'left')
        assert_equal(b, out)
        b = a.searchsorted(a, 'right')
        assert_equal(b, out + 1)
        e = np.ndarray(shape=0, buffer=b'', dtype=dt)
        b = e.searchsorted(a, 'left')
        assert_array_equal(b, np.zeros(len(a), dtype=np.intp))
        b = a.searchsorted(e, 'left')
        assert_array_equal(b, np.zeros(0, dtype=np.intp))