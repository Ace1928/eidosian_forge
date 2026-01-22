from functools import partial
from itertools import product
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
from scipy._lib._util import np_long
class TestExpmActionSimple:
    """
    These tests do not consider the case of multiple time steps in one call.
    """

    def test_theta_monotonicity(self):
        pairs = sorted(_theta.items())
        for (m_a, theta_a), (m_b, theta_b) in zip(pairs[:-1], pairs[1:]):
            assert_(theta_a < theta_b)

    def test_p_max_default(self):
        m_max = 55
        expected_p_max = 8
        observed_p_max = _compute_p_max(m_max)
        assert_equal(observed_p_max, expected_p_max)

    def test_p_max_range(self):
        for m_max in range(1, 55 + 1):
            p_max = _compute_p_max(m_max)
            assert_(p_max * (p_max - 1) <= m_max + 1)
            p_too_big = p_max + 1
            assert_(p_too_big * (p_too_big - 1) > m_max + 1)

    def test_onenormest_matrix_power(self):
        np.random.seed(1234)
        n = 40
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            for p in range(4):
                if not p:
                    M = np.identity(n)
                else:
                    M = np.dot(M, A)
                estimated = _onenormest_matrix_power(A, p)
                exact = np.linalg.norm(M, 1)
                assert_(less_than_or_close(estimated, exact))
                assert_(less_than_or_close(exact, 3 * estimated))

    def test_expm_multiply(self):
        np.random.seed(1234)
        n = 40
        k = 3
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            B = np.random.randn(n, k)
            observed = expm_multiply(A, B)
            expected = np.dot(sp_expm(A), B)
            assert_allclose(observed, expected)
            observed = estimated(expm_multiply)(aslinearoperator(A), B)
            assert_allclose(observed, expected)
            traceA = np.trace(A)
            observed = expm_multiply(aslinearoperator(A), B, traceA=traceA)
            assert_allclose(observed, expected)

    def test_matrix_vector_multiply(self):
        np.random.seed(1234)
        n = 40
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            v = np.random.randn(n)
            observed = expm_multiply(A, v)
            expected = np.dot(sp_expm(A), v)
            assert_allclose(observed, expected)
            observed = estimated(expm_multiply)(aslinearoperator(A), v)
            assert_allclose(observed, expected)

    def test_scaled_expm_multiply(self):
        np.random.seed(1234)
        n = 40
        k = 3
        nsamples = 10
        for i, t in product(range(nsamples), [0.2, 1.0, 1.5]):
            with np.errstate(invalid='ignore'):
                A = scipy.linalg.inv(np.random.randn(n, n))
                B = np.random.randn(n, k)
                observed = _expm_multiply_simple(A, B, t=t)
                expected = np.dot(sp_expm(t * A), B)
                assert_allclose(observed, expected)
                observed = estimated(_expm_multiply_simple)(aslinearoperator(A), B, t=t)
                assert_allclose(observed, expected)

    def test_scaled_expm_multiply_single_timepoint(self):
        np.random.seed(1234)
        t = 0.1
        n = 5
        k = 2
        A = np.random.randn(n, n)
        B = np.random.randn(n, k)
        observed = _expm_multiply_simple(A, B, t=t)
        expected = sp_expm(t * A).dot(B)
        assert_allclose(observed, expected)
        observed = estimated(_expm_multiply_simple)(aslinearoperator(A), B, t=t)
        assert_allclose(observed, expected)

    def test_sparse_expm_multiply(self):
        np.random.seed(1234)
        n = 40
        k = 3
        nsamples = 10
        for i in range(nsamples):
            A = scipy.sparse.rand(n, n, density=0.05)
            B = np.random.randn(n, k)
            observed = expm_multiply(A, B)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'splu converted its input to CSC format')
                sup.filter(SparseEfficiencyWarning, 'spsolve is more efficient when sparse b is in the CSC matrix format')
                expected = sp_expm(A).dot(B)
            assert_allclose(observed, expected)
            observed = estimated(expm_multiply)(aslinearoperator(A), B)
            assert_allclose(observed, expected)

    def test_complex(self):
        A = np.array([[1j, 1j], [0, 1j]], dtype=complex)
        B = np.array([1j, 1j])
        observed = expm_multiply(A, B)
        expected = np.array([1j * np.exp(1j) + 1j * (1j * np.cos(1) - np.sin(1)), 1j * np.exp(1j)], dtype=complex)
        assert_allclose(observed, expected)
        observed = estimated(expm_multiply)(aslinearoperator(A), B)
        assert_allclose(observed, expected)