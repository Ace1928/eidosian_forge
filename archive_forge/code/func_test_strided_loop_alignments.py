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
def test_strided_loop_alignments(self):
    for align in [1, 2, 4, 8, 12, 16, None]:
        xf64 = _aligned_zeros(3, np.float64)
        xc64 = _aligned_zeros(3, np.complex64, align=align)
        xf128 = _aligned_zeros(3, np.longdouble, align=align)
        with suppress_warnings() as sup:
            sup.filter(np.ComplexWarning, 'Casting complex values')
            xc64.astype('f8')
        xf64.astype(np.complex64)
        test = xc64 + xf64
        xf128.astype('f8')
        xf64.astype(np.longdouble)
        test = xf128 + xf64
        test = xf128 + xc64
        xf64[:] = xf64.copy()
        xc64[:] = xc64.copy()
        xf128[:] = xf128.copy()
        xf64[::2] = xf64[::2].copy()
        xc64[::2] = xc64[::2].copy()
        xf128[::2] = xf128[::2].copy()