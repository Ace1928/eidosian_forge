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
@pytest.mark.valgrind_error(reason='leaks buffer info cache temporarily.')
@pytest.mark.skipif(not np.ones((10, 1), order='C').flags.f_contiguous, reason='Test is unnecessary (but fails) without relaxed strides.')
def test_relaxed_strides_buffer_info_leak(self, arr=np.ones((1, 10))):
    """Test that alternating export of C- and F-order buffers from
        an array which is both C- and F-order when relaxed strides is
        active works.
        This test defines array in the signature to ensure leaking more
        references every time the test is run (catching the leak with
        pytest-leaks).
        """
    for i in range(10):
        _, s = _multiarray_tests.get_buffer_info(arr, ['F_CONTIGUOUS'])
        assert s == (8, 8)
        _, s = _multiarray_tests.get_buffer_info(arr, ['C_CONTIGUOUS'])
        assert s == (80, 8)