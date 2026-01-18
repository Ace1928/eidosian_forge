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
def test_vector_vector_values(self):
    vec1 = np.array([1, 2])
    vec2 = np.array([3, 4]).reshape(-1, 1)
    tgt1 = np.array([11])
    tgt2 = np.array([[3, 6], [4, 8]])
    for dt in self.types[1:]:
        v1 = vec1.astype(dt)
        v2 = vec2.astype(dt)
        res = self.matmul(v1, v2)
        assert_equal(res, tgt1)
        res = self.matmul(v2, v1.reshape(1, -1))
        assert_equal(res, tgt2)
    vec = np.array([True, True], dtype='?')
    res = self.matmul(vec, vec)
    assert_equal(res, True)