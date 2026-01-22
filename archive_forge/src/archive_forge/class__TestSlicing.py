import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
class _TestSlicing:

    def test_dtype_preservation(self):
        assert_equal(self.spcreator((1, 10), dtype=np.int16)[0, 1:5].dtype, np.int16)
        assert_equal(self.spcreator((1, 10), dtype=np.int32)[0, 1:5].dtype, np.int32)
        assert_equal(self.spcreator((1, 10), dtype=np.float32)[0, 1:5].dtype, np.float32)
        assert_equal(self.spcreator((1, 10), dtype=np.float64)[0, 1:5].dtype, np.float64)

    def test_dtype_preservation_empty_slice(self):
        for dt in [np.int16, np.int32, np.float32, np.float64]:
            A = self.spcreator((3, 2), dtype=dt)
            assert_equal(A[:, 0:0:2].dtype, dt)
            assert_equal(A[0:0:2, :].dtype, dt)
            assert_equal(A[0, 0:0:2].dtype, dt)
            assert_equal(A[0:0:2, 0].dtype, dt)

    def test_get_horiz_slice(self):
        B = asmatrix(arange(50.0).reshape(5, 10))
        A = self.spcreator(B)
        assert_array_equal(B[1, :], A[1, :].toarray())
        assert_array_equal(B[1, 2:5], A[1, 2:5].toarray())
        C = matrix([[1, 2, 1], [4, 0, 6], [0, 0, 0], [0, 0, 1]])
        D = self.spcreator(C)
        assert_array_equal(C[1, 1:3], D[1, 1:3].toarray())
        E = matrix([[1, 2, 1], [4, 0, 0], [0, 0, 0], [0, 0, 1]])
        F = self.spcreator(E)
        assert_array_equal(E[1, 1:3], F[1, 1:3].toarray())
        assert_array_equal(E[2, -2:], F[2, -2:].toarray())
        assert_raises(IndexError, A.__getitem__, (slice(None), 11))
        assert_raises(IndexError, A.__getitem__, (6, slice(3, 7)))

    def test_get_vert_slice(self):
        B = arange(50.0).reshape(5, 10)
        A = self.spcreator(B)
        assert_array_equal(B[2:5, [0]], A[2:5, 0].toarray())
        assert_array_equal(B[:, [1]], A[:, 1].toarray())
        C = array([[1, 2, 1], [4, 0, 6], [0, 0, 0], [0, 0, 1]])
        D = self.spcreator(C)
        assert_array_equal(C[1:3, [1]], D[1:3, 1].toarray())
        assert_array_equal(C[:, [2]], D[:, 2].toarray())
        E = array([[1, 0, 1], [4, 0, 0], [0, 0, 0], [0, 0, 1]])
        F = self.spcreator(E)
        assert_array_equal(E[:, [1]], F[:, 1].toarray())
        assert_array_equal(E[-2:, [2]], F[-2:, 2].toarray())
        assert_raises(IndexError, A.__getitem__, (slice(None), 11))
        assert_raises(IndexError, A.__getitem__, (6, slice(3, 7)))

    def test_get_slices(self):
        B = arange(50.0).reshape(5, 10)
        A = self.spcreator(B)
        assert_array_equal(A[2:5, 0:3].toarray(), B[2:5, 0:3])
        assert_array_equal(A[1:, :-1].toarray(), B[1:, :-1])
        assert_array_equal(A[:-1, 1:].toarray(), B[:-1, 1:])
        E = array([[1, 0, 1], [4, 0, 0], [0, 0, 0], [0, 0, 1]])
        F = self.spcreator(E)
        assert_array_equal(E[1:2, 1:2], F[1:2, 1:2].toarray())
        assert_array_equal(E[:, 1:], F[:, 1:].toarray())

    def test_non_unit_stride_2d_indexing(self):
        v0 = np.random.rand(50, 50)
        try:
            v = self.spcreator(v0)[0:25:2, 2:30:3]
        except ValueError:
            raise pytest.skip('feature not implemented')
        assert_array_equal(v.toarray(), v0[0:25:2, 2:30:3])

    def test_slicing_2(self):
        B = asmatrix(arange(50).reshape(5, 10))
        A = self.spcreator(B)
        assert_equal(A[2, 3], B[2, 3])
        assert_equal(A[-1, 8], B[-1, 8])
        assert_equal(A[-1, -2], B[-1, -2])
        assert_equal(A[array(-1), -2], B[-1, -2])
        assert_equal(A[-1, array(-2)], B[-1, -2])
        assert_equal(A[array(-1), array(-2)], B[-1, -2])
        assert_equal(A[2, :].toarray(), B[2, :])
        assert_equal(A[2, 5:-2].toarray(), B[2, 5:-2])
        assert_equal(A[array(2), 5:-2].toarray(), B[2, 5:-2])
        assert_equal(A[:, 2].toarray(), B[:, 2])
        assert_equal(A[3:4, 9].toarray(), B[3:4, 9])
        assert_equal(A[1:4, -5].toarray(), B[1:4, -5])
        assert_equal(A[2:-1, 3].toarray(), B[2:-1, 3])
        assert_equal(A[2:-1, array(3)].toarray(), B[2:-1, 3])
        assert_equal(A[1:2, 1:2].toarray(), B[1:2, 1:2])
        assert_equal(A[4:, 3:].toarray(), B[4:, 3:])
        assert_equal(A[:4, :5].toarray(), B[:4, :5])
        assert_equal(A[2:-1, :5].toarray(), B[2:-1, :5])
        assert_equal(A[1, :].toarray(), B[1, :])
        assert_equal(A[-2, :].toarray(), B[-2, :])
        assert_equal(A[array(-2), :].toarray(), B[-2, :])
        assert_equal(A[1:4].toarray(), B[1:4])
        assert_equal(A[1:-2].toarray(), B[1:-2])
        s = slice(int8(2), int8(4), None)
        assert_equal(A[s, :].toarray(), B[2:4, :])
        assert_equal(A[:, s].toarray(), B[:, 2:4])

    def test_slicing_3(self):
        B = asmatrix(arange(50).reshape(5, 10))
        A = self.spcreator(B)
        s_ = np.s_
        slices = [s_[:2], s_[1:2], s_[3:], s_[3::2], s_[15:20], s_[3:2], s_[8:3:-1], s_[4::-2], s_[:5:-1], 0, 1, s_[:], s_[1:5], -1, -2, -5, array(-1), np.int8(-3)]

        def check_1(a):
            x = A[a]
            y = B[a]
            if y.shape == ():
                assert_equal(x, y, repr(a))
            elif x.size == 0 and y.size == 0:
                pass
            else:
                assert_array_equal(x.toarray(), y, repr(a))
        for j, a in enumerate(slices):
            check_1(a)

        def check_2(a, b):
            if isinstance(a, np.ndarray):
                ai = int(a)
            else:
                ai = a
            if isinstance(b, np.ndarray):
                bi = int(b)
            else:
                bi = b
            x = A[a, b]
            y = B[ai, bi]
            if y.shape == ():
                assert_equal(x, y, repr((a, b)))
            elif x.size == 0 and y.size == 0:
                pass
            else:
                assert_array_equal(x.toarray(), y, repr((a, b)))
        for i, a in enumerate(slices):
            for j, b in enumerate(slices):
                check_2(a, b)
        extra_slices = []
        for a, b, c in itertools.product(*[(None, 0, 1, 2, 5, 15, -1, -2, 5, -15)] * 3):
            if c == 0:
                continue
            extra_slices.append(slice(a, b, c))
        for a in extra_slices:
            check_2(a, a)
            check_2(a, -2)
            check_2(-2, a)

    def test_ellipsis_slicing(self):
        b = asmatrix(arange(50).reshape(5, 10))
        a = self.spcreator(b)
        assert_array_equal(a[...].toarray(), b[...].A)
        assert_array_equal(a[...,].toarray(), b[...,].A)
        assert_array_equal(a[1, ...].toarray(), b[1, ...].A)
        assert_array_equal(a[..., 1].toarray(), b[..., 1].A)
        assert_array_equal(a[1:, ...].toarray(), b[1:, ...].A)
        assert_array_equal(a[..., 1:].toarray(), b[..., 1:].A)
        assert_array_equal(a[1:, 1, ...].toarray(), b[1:, 1, ...].A)
        assert_array_equal(a[1, ..., 1:].toarray(), b[1, ..., 1:].A)
        assert_equal(a[1, 1, ...], b[1, 1, ...])
        assert_equal(a[1, ..., 1], b[1, ..., 1])

    def test_multiple_ellipsis_slicing(self):
        b = asmatrix(arange(50).reshape(5, 10))
        a = self.spcreator(b)
        with pytest.deprecated_call(match='removed in v1.13'):
            assert_array_equal(a[..., ...].toarray(), b[:, :].A)
        with pytest.deprecated_call(match='removed in v1.13'):
            assert_array_equal(a[..., ..., ...].toarray(), b[:, :].A)
        with pytest.deprecated_call(match='removed in v1.13'):
            assert_array_equal(a[1, ..., ...].toarray(), b[1, :].A)
        with pytest.deprecated_call(match='removed in v1.13'):
            assert_array_equal(a[1:, ..., ...].toarray(), b[1:, :].A)
        with pytest.deprecated_call(match='removed in v1.13'):
            assert_array_equal(a[..., ..., 1:].toarray(), b[:, 1:].A)
        with pytest.deprecated_call(match='removed in v1.13'):
            assert_array_equal(a[..., ..., 1].toarray(), b[:, 1].A)