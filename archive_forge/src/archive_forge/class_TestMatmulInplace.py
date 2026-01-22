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
class TestMatmulInplace:
    DTYPES = {}
    for i in MatmulCommon.types:
        for j in MatmulCommon.types:
            if np.can_cast(j, i):
                DTYPES[f'{i}-{j}'] = (np.dtype(i), np.dtype(j))

    @pytest.mark.parametrize('dtype1,dtype2', DTYPES.values(), ids=DTYPES)
    def test_basic(self, dtype1: np.dtype, dtype2: np.dtype) -> None:
        a = np.arange(10).reshape(5, 2).astype(dtype1)
        a_id = id(a)
        b = np.ones((2, 2), dtype=dtype2)
        ref = a @ b
        a @= b
        assert id(a) == a_id
        assert a.dtype == dtype1
        assert a.shape == (5, 2)
        if dtype1.kind in 'fc':
            np.testing.assert_allclose(a, ref)
        else:
            np.testing.assert_array_equal(a, ref)
    SHAPES = {'2d_large': ((10 ** 5, 10), (10, 10)), '3d_large': ((10 ** 4, 10, 10), (1, 10, 10)), '1d': ((3,), (3,)), '2d_1d': ((3, 3), (3,)), '1d_2d': ((3,), (3, 3)), '2d_broadcast': ((3, 3), (3, 1)), '2d_broadcast_reverse': ((1, 3), (3, 3)), '3d_broadcast1': ((3, 3, 3), (1, 3, 1)), '3d_broadcast2': ((3, 3, 3), (1, 3, 3)), '3d_broadcast3': ((3, 3, 3), (3, 3, 1)), '3d_broadcast_reverse1': ((1, 3, 3), (3, 3, 3)), '3d_broadcast_reverse2': ((3, 1, 3), (3, 3, 3)), '3d_broadcast_reverse3': ((1, 1, 3), (3, 3, 3))}

    @pytest.mark.parametrize('a_shape,b_shape', SHAPES.values(), ids=SHAPES)
    def test_shapes(self, a_shape: tuple[int, ...], b_shape: tuple[int, ...]):
        a_size = np.prod(a_shape)
        a = np.arange(a_size).reshape(a_shape).astype(np.float64)
        a_id = id(a)
        b_size = np.prod(b_shape)
        b = np.arange(b_size).reshape(b_shape)
        ref = a @ b
        if ref.shape != a_shape:
            with pytest.raises(ValueError):
                a @= b
            return
        else:
            a @= b
        assert id(a) == a_id
        assert a.dtype.type == np.float64
        assert a.shape == a_shape
        np.testing.assert_allclose(a, ref)