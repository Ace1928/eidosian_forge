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
class TestMatmul(MatmulCommon):
    matmul = np.matmul

    def test_out_arg(self):
        a = np.ones((5, 2), dtype=float)
        b = np.array([[1, 3], [5, 7]], dtype=float)
        tgt = np.dot(a, b)
        msg = 'out positional argument'
        out = np.zeros((5, 2), dtype=float)
        self.matmul(a, b, out)
        assert_array_equal(out, tgt, err_msg=msg)
        msg = 'out keyword argument'
        out = np.zeros((5, 2), dtype=float)
        self.matmul(a, b, out=out)
        assert_array_equal(out, tgt, err_msg=msg)
        msg = 'Cannot cast ufunc .* output'
        out = np.zeros((5, 2), dtype=np.int32)
        assert_raises_regex(TypeError, msg, self.matmul, a, b, out=out)
        out = np.zeros((5, 2), dtype=np.complex128)
        c = self.matmul(a, b, out=out)
        assert_(c is out)
        with suppress_warnings() as sup:
            sup.filter(np.ComplexWarning, '')
            c = c.astype(tgt.dtype)
        assert_array_equal(c, tgt)

    def test_empty_out(self):
        arr = np.ones((0, 1, 1))
        out = np.ones((1, 1, 1))
        assert self.matmul(arr, arr).shape == (0, 1, 1)
        with pytest.raises(ValueError, match='non-broadcastable'):
            self.matmul(arr, arr, out=out)

    def test_out_contiguous(self):
        a = np.ones((5, 2), dtype=float)
        b = np.array([[1, 3], [5, 7]], dtype=float)
        v = np.array([1, 3], dtype=float)
        tgt = np.dot(a, b)
        tgt_mv = np.dot(a, v)
        out = np.ones((5, 2, 2), dtype=float)
        c = self.matmul(a, b, out=out[..., 0])
        assert c.base is out
        assert_array_equal(c, tgt)
        c = self.matmul(a, v, out=out[:, 0, 0])
        assert_array_equal(c, tgt_mv)
        c = self.matmul(v, a.T, out=out[:, 0, 0])
        assert_array_equal(c, tgt_mv)
        out = np.ones((10, 2), dtype=float)
        c = self.matmul(a, b, out=out[::2, :])
        assert_array_equal(c, tgt)
        out = np.ones((5, 2), dtype=float)
        c = self.matmul(b.T, a.T, out=out.T)
        assert_array_equal(out, tgt)
    m1 = np.arange(15.0).reshape(5, 3)
    m2 = np.arange(21.0).reshape(3, 7)
    m3 = np.arange(30.0).reshape(5, 6)[:, ::2]
    vc = np.arange(10.0)
    vr = np.arange(6.0)
    m0 = np.zeros((3, 0))

    @pytest.mark.parametrize('args', ((m1, m2), (m2.T, m1.T), (m2.T.copy(), m1.T), (m2.T, m1.T.copy()), (m1, m1.T), (m1.T, m1), (m1, m3.T), (m3, m1.T), (m3, m3.T), (m3.T, m3), (m3, m2), (m2.T, m3.T), (m2.T.copy(), m3.T), (m1, vr[:3]), (vc[:5], m1), (m1.T, vc[:5]), (vr[:3], m1.T), (m1, vr[::2]), (vc[::2], m1), (m1.T, vc[::2]), (vr[::2], m1.T), (m3, vr[:3]), (vc[:5], m3), (m3.T, vc[:5]), (vr[:3], m3.T), (m3, vr[::2]), (vc[::2], m3), (m3.T, vc[::2]), (vr[::2], m3.T), (m0, m0.T), (m0.T, m0), (m1, m0), (m0.T, m1.T)))
    def test_dot_equivalent(self, args):
        r1 = np.matmul(*args)
        r2 = np.dot(*args)
        assert_equal(r1, r2)
        r3 = np.matmul(args[0].copy(), args[1].copy())
        assert_equal(r1, r3)

    def test_matmul_object(self):
        import fractions
        f = np.vectorize(fractions.Fraction)

        def random_ints():
            return np.random.randint(1, 1000, size=(10, 3, 3))
        M1 = f(random_ints(), random_ints())
        M2 = f(random_ints(), random_ints())
        M3 = self.matmul(M1, M2)
        [N1, N2, N3] = [a.astype(float) for a in [M1, M2, M3]]
        assert_allclose(N3, self.matmul(N1, N2))

    def test_matmul_object_type_scalar(self):
        from fractions import Fraction as F
        v = np.array([F(2, 3), F(5, 7)])
        res = self.matmul(v, v)
        assert_(type(res) is F)

    def test_matmul_empty(self):
        a = np.empty((3, 0), dtype=object)
        b = np.empty((0, 3), dtype=object)
        c = np.zeros((3, 3))
        assert_array_equal(np.matmul(a, b), c)

    def test_matmul_exception_multiply(self):

        class add_not_multiply:

            def __add__(self, other):
                return self
        a = np.full((3, 3), add_not_multiply())
        with assert_raises(TypeError):
            b = np.matmul(a, a)

    def test_matmul_exception_add(self):

        class multiply_not_add:

            def __mul__(self, other):
                return self
        a = np.full((3, 3), multiply_not_add())
        with assert_raises(TypeError):
            b = np.matmul(a, a)

    def test_matmul_bool(self):
        a = np.array([[1, 0], [1, 1]], dtype=bool)
        assert np.max(a.view(np.uint8)) == 1
        b = np.matmul(a, a)
        assert np.max(b.view(np.uint8)) == 1
        rg = np.random.default_rng(np.random.PCG64(43))
        d = rg.integers(2, size=4 * 5, dtype=np.int8)
        d = d.reshape(4, 5) > 0
        out1 = np.matmul(d, d.reshape(5, 4))
        out2 = np.dot(d, d.reshape(5, 4))
        assert_equal(out1, out2)
        c = np.matmul(np.zeros((2, 0), dtype=bool), np.zeros(0, dtype=bool))
        assert not np.any(c)