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
class TestInvwishart:

    def test_frozen(self):
        dim = 4
        scale = np.diag(np.arange(dim) + 1)
        scale[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) / 2)
        scale = np.dot(scale.T, scale)
        X = []
        for i in range(5):
            x = np.diag(np.arange(dim) + (i + 1) ** 2)
            x[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) / 2)
            x = np.dot(x.T, x)
            X.append(x)
        X = np.array(X).T
        parameters = [(10, 1, np.linspace(0.1, 10, 5)), (10, scale, X)]
        for df, scale, x in parameters:
            iw = invwishart(df, scale)
            assert_equal(iw.var(), invwishart.var(df, scale))
            assert_equal(iw.mean(), invwishart.mean(df, scale))
            assert_equal(iw.mode(), invwishart.mode(df, scale))
            assert_allclose(iw.pdf(x), invwishart.pdf(x, df, scale))

    def test_1D_is_invgamma(self):
        np.random.seed(482974)
        sn = 500
        dim = 1
        scale = np.eye(dim)
        df_range = np.arange(5, 20, 2, dtype=float)
        X = np.linspace(0.1, 10, num=10)
        for df in df_range:
            iw = invwishart(df, scale)
            ig = invgamma(df / 2, scale=1.0 / 2)
            assert_allclose(iw.var(), ig.var())
            assert_allclose(iw.mean(), ig.mean())
            assert_allclose(iw.pdf(X), ig.pdf(X))
            rvs = iw.rvs(size=sn)
            args = (df / 2, 0, 1.0 / 2)
            alpha = 0.01
            check_distribution_rvs('invgamma', args, alpha, rvs)
            assert_allclose(iw.entropy(), ig.entropy())

    def test_invwishart_2D_rvs(self):
        dim = 3
        df = 10
        scale = np.eye(dim)
        scale[0, 1] = 0.5
        scale[1, 0] = 0.5
        iw = invwishart(df, scale)
        np.random.seed(608072)
        iw_rvs = invwishart.rvs(df, scale)
        np.random.seed(608072)
        frozen_iw_rvs = iw.rvs()
        np.random.seed(608072)
        covariances = np.random.normal(size=3)
        variances = np.r_[np.random.chisquare(df - 2), np.random.chisquare(df - 1), np.random.chisquare(df)] ** 0.5
        A = np.diag(variances)
        A[np.tril_indices(dim, k=-1)] = covariances
        D = np.linalg.cholesky(scale)
        L = np.linalg.solve(A.T, D.T).T
        manual_iw_rvs = np.dot(L, L.T)
        assert_allclose(iw_rvs, manual_iw_rvs)
        assert_allclose(frozen_iw_rvs, manual_iw_rvs)

    def test_sample_mean(self):
        """Test that sample mean consistent with known mean."""
        df = 10
        sample_size = 20000
        for dim in [1, 5]:
            scale = np.diag(np.arange(dim) + 1)
            scale[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) / 2)
            scale = np.dot(scale.T, scale)
            dist = invwishart(df, scale)
            Xmean_exp = dist.mean()
            Xvar_exp = dist.var()
            Xmean_std = (Xvar_exp / sample_size) ** 0.5
            X = dist.rvs(size=sample_size, random_state=1234)
            Xmean_est = X.mean(axis=0)
            ntests = dim * (dim + 1) // 2
            fail_rate = 0.01 / ntests
            max_diff = norm.ppf(1 - fail_rate / 2)
            assert np.allclose((Xmean_est - Xmean_exp) / Xmean_std, 0, atol=max_diff)

    def test_logpdf_4x4(self):
        """Regression test for gh-8844."""
        X = np.array([[2, 1, 0, 0.5], [1, 2, 0.5, 0.5], [0, 0.5, 3, 1], [0.5, 0.5, 1, 2]])
        Psi = np.array([[9, 7, 3, 1], [7, 9, 5, 1], [3, 5, 8, 2], [1, 1, 2, 9]])
        nu = 6
        prob = invwishart.logpdf(X, nu, Psi)
        p = X.shape[0]
        sig, logdetX = np.linalg.slogdet(X)
        sig, logdetPsi = np.linalg.slogdet(Psi)
        M = np.linalg.solve(X, Psi)
        expected = nu / 2 * logdetPsi - nu * p / 2 * np.log(2) - multigammaln(nu / 2, p) - (nu + p + 1) / 2 * logdetX - 0.5 * M.trace()
        assert_allclose(prob, expected)