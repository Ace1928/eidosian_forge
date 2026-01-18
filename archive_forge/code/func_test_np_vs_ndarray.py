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
@pytest.mark.parametrize('arr_method, np_method', [('argmax', np.argmax), ('argmin', np.argmin)])
def test_np_vs_ndarray(self, arr_method, np_method):
    a = np.random.normal(size=(2, 3))
    arg_method = getattr(a, arr_method)
    out1 = np.zeros(2, dtype=int)
    out2 = np.zeros(2, dtype=int)
    assert_equal(arg_method(1, out1), np_method(a, 1, out2))
    assert_equal(out1, out2)
    out1 = np.zeros(3, dtype=int)
    out2 = np.zeros(3, dtype=int)
    assert_equal(arg_method(out=out1, axis=0), np_method(a, out=out2, axis=0))
    assert_equal(out1, out2)