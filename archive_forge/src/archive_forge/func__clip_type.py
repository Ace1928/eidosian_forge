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
def _clip_type(self, type_group, array_max, clip_min, clip_max, inplace=False, expected_min=None, expected_max=None):
    if expected_min is None:
        expected_min = clip_min
    if expected_max is None:
        expected_max = clip_max
    for T in np.sctypes[type_group]:
        if sys.byteorder == 'little':
            byte_orders = ['=', '>']
        else:
            byte_orders = ['<', '=']
        for byteorder in byte_orders:
            dtype = np.dtype(T).newbyteorder(byteorder)
            x = (np.random.random(1000) * array_max).astype(dtype)
            if inplace:
                x.clip(clip_min, clip_max, x, casting='unsafe')
            else:
                x = x.clip(clip_min, clip_max)
                byteorder = '='
            if x.dtype.byteorder == '|':
                byteorder = '|'
            assert_equal(x.dtype.byteorder, byteorder)
            self._check_range(x, expected_min, expected_max)
    return x