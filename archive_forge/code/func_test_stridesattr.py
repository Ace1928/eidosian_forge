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
def test_stridesattr(self):
    x = self.one

    def make_array(size, offset, strides):
        return np.ndarray(size, buffer=x, dtype=int, offset=offset * x.itemsize, strides=strides * x.itemsize)
    assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
    assert_raises(ValueError, make_array, 4, 4, -2)
    assert_raises(ValueError, make_array, 4, 2, -1)
    assert_raises(ValueError, make_array, 8, 3, 1)
    assert_equal(make_array(8, 3, 0), np.array([3] * 8))
    assert_raises(ValueError, make_array, (2, 3), 5, np.array([-2, -3]))
    make_array(0, 0, 10)