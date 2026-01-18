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
def test_uintalignment_and_alignment():
    d1 = np.dtype('u1,c8', align=True)
    d2 = np.dtype('u4,c8', align=True)
    d3 = np.dtype({'names': ['a', 'b'], 'formats': ['u1', d1]}, align=True)
    assert_equal(np.zeros(1, dtype=d1)['f1'].flags['ALIGNED'], True)
    assert_equal(np.zeros(1, dtype=d2)['f1'].flags['ALIGNED'], True)
    assert_equal(np.zeros(1, dtype='u1,c8')['f1'].flags['ALIGNED'], False)
    s = _multiarray_tests.get_struct_alignments()
    for d, (alignment, size) in zip([d1, d2, d3], s):
        assert_equal(d.alignment, alignment)
        assert_equal(d.itemsize, size)
    src = np.zeros((2, 2), dtype=d1)['f1']
    np.exp(src)
    dst = np.zeros((2, 2), dtype='c8')
    dst[:, 1] = src[:, 1]