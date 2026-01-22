import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
class TestMatrixNormal:

    def test_bad_input(self):
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        assert_raises(ValueError, matrix_normal, np.zeros((5, 4, 3)))
        assert_raises(ValueError, matrix_normal, M, np.zeros(10), V)
        assert_raises(ValueError, matrix_normal, M, U, np.zeros(10))
        assert_raises(ValueError, matrix_normal, M, U, U)
        assert_raises(ValueError, matrix_normal, M, V, V)
        assert_raises(ValueError, matrix_normal, M.T, U, V)
        e = np.linalg.LinAlgError
        assert_raises(e, matrix_normal.rvs, M, U, np.ones((num_cols, num_cols)))
        assert_raises(e, matrix_normal.rvs, M, np.ones((num_rows, num_rows)), V)
        assert_raises(e, matrix_normal, M, U, np.ones((num_cols, num_cols)))
        assert_raises(e, matrix_normal, M, np.ones((num_rows, num_rows)), V)

    def test_default_inputs(self):
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        Z = np.zeros((num_rows, num_cols))
        Zr = np.zeros((num_rows, 1))
        Zc = np.zeros((1, num_cols))
        Ir = np.identity(num_rows)
        Ic = np.identity(num_cols)
        I1 = np.identity(1)
        assert_equal(matrix_normal.rvs(mean=M, rowcov=U, colcov=V).shape, (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(mean=M).shape, (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(rowcov=U).shape, (num_rows, 1))
        assert_equal(matrix_normal.rvs(colcov=V).shape, (1, num_cols))
        assert_equal(matrix_normal.rvs(mean=M, colcov=V).shape, (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(mean=M, rowcov=U).shape, (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(rowcov=U, colcov=V).shape, (num_rows, num_cols))
        assert_equal(matrix_normal(mean=M).rowcov, Ir)
        assert_equal(matrix_normal(mean=M).colcov, Ic)
        assert_equal(matrix_normal(rowcov=U).mean, Zr)
        assert_equal(matrix_normal(rowcov=U).colcov, I1)
        assert_equal(matrix_normal(colcov=V).mean, Zc)
        assert_equal(matrix_normal(colcov=V).rowcov, I1)
        assert_equal(matrix_normal(mean=M, rowcov=U).colcov, Ic)
        assert_equal(matrix_normal(mean=M, colcov=V).rowcov, Ir)
        assert_equal(matrix_normal(rowcov=U, colcov=V).mean, Z)

    def test_covariance_expansion(self):
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        Uv = np.full(num_rows, 0.2)
        Us = 0.2
        Vv = np.full(num_cols, 0.1)
        Vs = 0.1
        Ir = np.identity(num_rows)
        Ic = np.identity(num_cols)
        assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).rowcov, 0.2 * Ir)
        assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).colcov, 0.1 * Ic)
        assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).rowcov, 0.2 * Ir)
        assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).colcov, 0.1 * Ic)

    def test_frozen_matrix_normal(self):
        for i in range(1, 5):
            for j in range(1, 5):
                M = np.full((i, j), 0.3)
                U = 0.5 * np.identity(i) + np.full((i, i), 0.5)
                V = 0.7 * np.identity(j) + np.full((j, j), 0.3)
                frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
                rvs1 = frozen.rvs(random_state=1234)
                rvs2 = matrix_normal.rvs(mean=M, rowcov=U, colcov=V, random_state=1234)
                assert_equal(rvs1, rvs2)
                X = frozen.rvs(random_state=1234)
                pdf1 = frozen.pdf(X)
                pdf2 = matrix_normal.pdf(X, mean=M, rowcov=U, colcov=V)
                assert_equal(pdf1, pdf2)
                logpdf1 = frozen.logpdf(X)
                logpdf2 = matrix_normal.logpdf(X, mean=M, rowcov=U, colcov=V)
                assert_equal(logpdf1, logpdf2)

    def test_matches_multivariate(self):
        for i in range(1, 5):
            for j in range(1, 5):
                M = np.full((i, j), 0.3)
                U = 0.5 * np.identity(i) + np.full((i, i), 0.5)
                V = 0.7 * np.identity(j) + np.full((j, j), 0.3)
                frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
                X = frozen.rvs(random_state=1234)
                pdf1 = frozen.pdf(X)
                logpdf1 = frozen.logpdf(X)
                entropy1 = frozen.entropy()
                vecX = X.T.flatten()
                vecM = M.T.flatten()
                cov = np.kron(V, U)
                pdf2 = multivariate_normal.pdf(vecX, mean=vecM, cov=cov)
                logpdf2 = multivariate_normal.logpdf(vecX, mean=vecM, cov=cov)
                entropy2 = multivariate_normal.entropy(mean=vecM, cov=cov)
                assert_allclose(pdf1, pdf2, rtol=1e-10)
                assert_allclose(logpdf1, logpdf2, rtol=1e-10)
                assert_allclose(entropy1, entropy2)

    def test_array_input(self):
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        N = 10
        frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
        X1 = frozen.rvs(size=N, random_state=1234)
        X2 = frozen.rvs(size=N, random_state=4321)
        X = np.concatenate((X1[np.newaxis, :, :, :], X2[np.newaxis, :, :, :]), axis=0)
        assert_equal(X.shape, (2, N, num_rows, num_cols))
        array_logpdf = frozen.logpdf(X)
        assert_equal(array_logpdf.shape, (2, N))
        for i in range(2):
            for j in range(N):
                separate_logpdf = matrix_normal.logpdf(X[i, j], mean=M, rowcov=U, colcov=V)
                assert_allclose(separate_logpdf, array_logpdf[i, j], 1e-10)

    def test_moments(self):
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        N = 1000
        frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
        X = frozen.rvs(size=N, random_state=1234)
        sample_mean = np.mean(X, axis=0)
        assert_allclose(sample_mean, M, atol=0.1)
        sample_colcov = np.cov(X.reshape(N * num_rows, num_cols).T)
        assert_allclose(sample_colcov, V, atol=0.1)
        sample_rowcov = np.cov(np.swapaxes(X, 1, 2).reshape(N * num_cols, num_rows).T)
        assert_allclose(sample_rowcov, U, atol=0.1)

    def test_samples(self):
        actual = matrix_normal.rvs(mean=np.array([[1, 2], [3, 4]]), rowcov=np.array([[4, -1], [-1, 2]]), colcov=np.array([[5, 1], [1, 10]]), random_state=np.random.default_rng(0), size=2)
        expected = np.array([[[1.56228264238181, -1.24136424071189], [2.46865788392114, 6.22964440489445]], [[3.86405716144353, 10.73714311429529], [2.59428444080606, 5.79987854490876]]])
        assert_allclose(actual, expected)