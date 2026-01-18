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
def test_record_array_with_object_dtype(self):
    my_object = object()
    arr_with_object = np.array([(my_object, 1, 2.0)], dtype=[('a', object), ('b', int), ('c', float)])
    arr_without_object = np.array([('xxx', 1, 2.0)], dtype=[('a', str), ('b', int), ('c', float)])
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        depickled_arr_with_object = pickle.loads(pickle.dumps(arr_with_object, protocol=proto))
        depickled_arr_without_object = pickle.loads(pickle.dumps(arr_without_object, protocol=proto))
        assert_equal(arr_with_object.dtype, depickled_arr_with_object.dtype)
        assert_equal(arr_without_object.dtype, depickled_arr_without_object.dtype)