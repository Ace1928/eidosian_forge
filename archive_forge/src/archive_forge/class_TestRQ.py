import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
class TestRQ:

    def test_simple(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        r, q = rq(a)
        assert_array_almost_equal(q @ q.T, eye(3))
        assert_array_almost_equal(r @ q, a)

    def test_r(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        r, q = rq(a)
        r2 = rq(a, mode='r')
        assert_array_almost_equal(r, r2)

    def test_random(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            r, q = rq(a)
            assert_array_almost_equal(q @ q.T, eye(n))
            assert_array_almost_equal(r @ q, a)

    def test_simple_trap(self):
        a = [[8, 2, 3], [2, 9, 3]]
        r, q = rq(a)
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(r @ q, a)

    def test_simple_tall(self):
        a = [[8, 2], [2, 9], [5, 3]]
        r, q = rq(a)
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(r @ q, a)

    def test_simple_fat(self):
        a = [[8, 2, 5], [2, 9, 3]]
        r, q = rq(a)
        assert_array_almost_equal(q @ q.T, eye(3))
        assert_array_almost_equal(r @ q, a)

    def test_simple_complex(self):
        a = [[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]]
        r, q = rq(a)
        assert_array_almost_equal(q @ q.conj().T, eye(3))
        assert_array_almost_equal(r @ q, a)

    def test_random_tall(self):
        rng = np.random.RandomState(1234)
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            r, q = rq(a)
            assert_array_almost_equal(q @ q.T, eye(n))
            assert_array_almost_equal(r @ q, a)

    def test_random_trap(self):
        rng = np.random.RandomState(1234)
        m = 100
        n = 200
        for k in range(2):
            a = rng.random([m, n])
            r, q = rq(a)
            assert_array_almost_equal(q @ q.T, eye(n))
            assert_array_almost_equal(r @ q, a)

    def test_random_trap_economic(self):
        rng = np.random.RandomState(1234)
        m = 100
        n = 200
        for k in range(2):
            a = rng.random([m, n])
            r, q = rq(a, mode='economic')
            assert_array_almost_equal(q @ q.T, eye(m))
            assert_array_almost_equal(r @ q, a)
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (m, m))

    def test_random_complex(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            r, q = rq(a)
            assert_array_almost_equal(q @ q.conj().T, eye(n))
            assert_array_almost_equal(r @ q, a)

    def test_random_complex_economic(self):
        rng = np.random.RandomState(1234)
        m = 100
        n = 200
        for k in range(2):
            a = rng.random([m, n]) + 1j * rng.random([m, n])
            r, q = rq(a, mode='economic')
            assert_array_almost_equal(q @ q.conj().T, eye(m))
            assert_array_almost_equal(r @ q, a)
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (m, m))

    def test_check_finite(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        r, q = rq(a, check_finite=False)
        assert_array_almost_equal(q @ q.T, eye(3))
        assert_array_almost_equal(r @ q, a)