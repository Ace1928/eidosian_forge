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
def test_vector_matrix_values(self):
    vec = np.array([1, 2])
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.stack([mat1] * 2, axis=0)
    tgt1 = np.array([7, 10])
    tgt2 = np.stack([tgt1] * 2, axis=0)
    for dt in self.types[1:]:
        v = vec.astype(dt)
        m1 = mat1.astype(dt)
        m2 = mat2.astype(dt)
        res = self.matmul(v, m1)
        assert_equal(res, tgt1)
        res = self.matmul(v, m2)
        assert_equal(res, tgt2)
    vec = np.array([True, False])
    mat1 = np.array([[True, False], [False, True]])
    mat2 = np.stack([mat1] * 2, axis=0)
    tgt1 = np.array([True, False])
    tgt2 = np.stack([tgt1] * 2, axis=0)
    res = self.matmul(vec, mat1)
    assert_equal(res, tgt1)
    res = self.matmul(vec, mat2)
    assert_equal(res, tgt2)