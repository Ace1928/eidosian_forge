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
@pytest.mark.parametrize('dt', [np.dtype([('f', float), ('i', int)]), np.dtype([('f', float), ('i', object)])])
@pytest.mark.parametrize('step', [1, 2])
def test_sort_structured(self, dt, step):
    a = np.array([(i, i) for i in range(101 * step)], dtype=dt)
    b = a[::-1]
    for kind in ['q', 'h', 'm']:
        msg = 'kind=%s' % kind
        c = a.copy()[::step]
        indx = c.argsort(kind=kind)
        c.sort(kind=kind)
        assert_equal(c, a[::step], msg)
        assert_equal(a[::step][indx], a[::step], msg)
        c = b.copy()[::step]
        indx = c.argsort(kind=kind)
        c.sort(kind=kind)
        assert_equal(c, a[step - 1::step], msg)
        assert_equal(b[::step][indx], a[step - 1::step], msg)