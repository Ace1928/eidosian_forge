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
@pytest.mark.leaks_references(reason='replaces None with NULL.')
@pytest.mark.parametrize('method, vals', [('argmax', (10, 30)), ('argmin', (30, 10))])
def test_object_with_NULLs(self, method, vals):
    a = np.empty(4, dtype='O')
    arg_method = getattr(a, method)
    ctypes.memset(a.ctypes.data, 0, a.nbytes)
    assert_equal(arg_method(), 0)
    a[3] = vals[0]
    assert_equal(arg_method(), 3)
    a[1] = vals[1]
    assert_equal(arg_method(), 1)