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
class TestCAPI:

    def test_IsPythonScalar(self):
        from numpy.core._multiarray_tests import IsPythonScalar
        assert_(IsPythonScalar(b'foobar'))
        assert_(IsPythonScalar(1))
        assert_(IsPythonScalar(2 ** 80))
        assert_(IsPythonScalar(2.0))
        assert_(IsPythonScalar('a'))

    @pytest.mark.parametrize('converter', [_multiarray_tests.run_scalar_intp_converter, _multiarray_tests.run_scalar_intp_from_sequence])
    def test_intp_sequence_converters(self, converter):
        assert converter(10) == (10,)
        assert converter(-1) == (-1,)
        assert converter(np.array(123)) == (123,)
        assert converter((10,)) == (10,)
        assert converter(np.array([11])) == (11,)

    @pytest.mark.parametrize('converter', [_multiarray_tests.run_scalar_intp_converter, _multiarray_tests.run_scalar_intp_from_sequence])
    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
    def test_intp_sequence_converters_errors(self, converter):
        with pytest.raises(TypeError, match='expected a sequence of integers or a single integer, '):
            converter(object())
        with pytest.raises(TypeError, match="expected a sequence of integers or a single integer, got '32.0'"):
            converter(32.0)
        with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
            converter([32.0])
        with pytest.raises(ValueError, match='Maximum allowed dimension'):
            converter(2 ** 64)