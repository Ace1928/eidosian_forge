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
@pytest.mark.valgrind_error(reason='leaks buffer info cache temporarily.')
def test_relaxed_strides(self, c=np.ones((1, 10, 10), dtype='i8')):
    c.strides = (-1, 80, 8)
    assert_(memoryview(c).strides == (800, 80, 8))
    fd = io.BytesIO()
    fd.write(c.data)
    fortran = c.T
    assert_(memoryview(fortran).strides == (8, 80, 800))
    arr = np.ones((1, 10))
    if arr.flags.f_contiguous:
        shape, strides = _multiarray_tests.get_buffer_info(arr, ['F_CONTIGUOUS'])
        assert_(strides[0] == 8)
        arr = np.ones((10, 1), order='F')
        shape, strides = _multiarray_tests.get_buffer_info(arr, ['C_CONTIGUOUS'])
        assert_(strides[-1] == 8)