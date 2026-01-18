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
def test_to_int_scalar(self):
    int_funcs = (int, lambda x: x.__int__())
    for int_func in int_funcs:
        assert_equal(int_func(np.array(0)), 0)
        with assert_warns(DeprecationWarning):
            assert_equal(int_func(np.array([1])), 1)
        with assert_warns(DeprecationWarning):
            assert_equal(int_func(np.array([[42]])), 42)
        assert_raises(TypeError, int_func, np.array([1, 2]))
        assert_equal(4, int_func(np.array('4')))
        assert_equal(5, int_func(np.bytes_(b'5')))
        assert_equal(6, int_func(np.str_('6')))
        if sys.version_info < (3, 11):

            class HasTrunc:

                def __trunc__(self):
                    return 3
            assert_equal(3, int_func(np.array(HasTrunc())))
            with assert_warns(DeprecationWarning):
                assert_equal(3, int_func(np.array([HasTrunc()])))
        else:
            pass

        class NotConvertible:

            def __int__(self):
                raise NotImplementedError
        assert_raises(NotImplementedError, int_func, np.array(NotConvertible()))
        with assert_warns(DeprecationWarning):
            assert_raises(NotImplementedError, int_func, np.array([NotConvertible()]))