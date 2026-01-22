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
class TestDirichlet:

    def test_frozen_dirichlet(self):
        np.random.seed(2846)
        n = np.random.randint(1, 32)
        alpha = np.random.uniform(1e-09, 100, n)
        d = dirichlet(alpha)
        assert_equal(d.var(), dirichlet.var(alpha))
        assert_equal(d.mean(), dirichlet.mean(alpha))
        assert_equal(d.entropy(), dirichlet.entropy(alpha))
        num_tests = 10
        for i in range(num_tests):
            x = np.random.uniform(1e-09, 100, n)
            x /= np.sum(x)
            assert_equal(d.pdf(x[:-1]), dirichlet.pdf(x[:-1], alpha))
            assert_equal(d.logpdf(x[:-1]), dirichlet.logpdf(x[:-1], alpha))

    def test_numpy_rvs_shape_compatibility(self):
        np.random.seed(2846)
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.random.dirichlet(alpha, size=7)
        assert_equal(x.shape, (7, 3))
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)
        dirichlet.pdf(x.T, alpha)
        dirichlet.pdf(x.T[:-1], alpha)
        dirichlet.logpdf(x.T, alpha)
        dirichlet.logpdf(x.T[:-1], alpha)

    def test_alpha_with_zeros(self):
        np.random.seed(2846)
        alpha = [1.0, 0.0, 3.0]
        x = np.random.dirichlet(np.maximum(1e-09, alpha), size=7).T
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_with_negative_entries(self):
        np.random.seed(2846)
        alpha = [1.0, -2.0, 3.0]
        x = np.random.dirichlet(np.maximum(1e-09, alpha), size=7).T
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_zeros(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, 0.0, 0.2, 0.7])
        dirichlet.pdf(x, alpha)
        dirichlet.logpdf(x, alpha)
        alpha = np.array([1.0, 1.0, 1.0, 1.0])
        assert_almost_equal(dirichlet.pdf(x, alpha), 6)
        assert_almost_equal(dirichlet.logpdf(x, alpha), np.log(6))

    def test_data_with_zeros_and_small_alpha(self):
        alpha = np.array([1.0, 0.5, 3.0, 4.0])
        x = np.array([0.1, 0.0, 0.2, 0.7])
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_negative_entries(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, -0.1, 0.3, 0.7])
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_too_large_entries(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, 1.1, 0.3, 0.7])
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_too_deep_c(self):
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.full((2, 7, 7), 1 / 14)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_too_deep(self):
        alpha = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.full((2, 2, 7), 1 / 4)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_correct_depth(self):
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.full((3, 7), 1 / 3)
        dirichlet.pdf(x, alpha)
        dirichlet.logpdf(x, alpha)

    def test_non_simplex_data(self):
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.full((3, 7), 1 / 2)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_vector_too_short(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.full((2, 7), 1 / 2)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_vector_too_long(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.full((5, 7), 1 / 5)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_mean_var_cov(self):
        alpha = np.array([1.0, 0.8, 0.2])
        d = dirichlet(alpha)
        expected_mean = [0.5, 0.4, 0.1]
        expected_var = [1.0 / 12.0, 0.08, 0.03]
        expected_cov = [[1.0 / 12, -1.0 / 15, -1.0 / 60], [-1.0 / 15, 2.0 / 25, -1.0 / 75], [-1.0 / 60, -1.0 / 75, 3.0 / 100]]
        assert_array_almost_equal(d.mean(), expected_mean)
        assert_array_almost_equal(d.var(), expected_var)
        assert_array_almost_equal(d.cov(), expected_cov)

    def test_scalar_values(self):
        alpha = np.array([0.2])
        d = dirichlet(alpha)
        assert_equal(d.mean().ndim, 0)
        assert_equal(d.var().ndim, 0)
        assert_equal(d.pdf([1.0]).ndim, 0)
        assert_equal(d.logpdf([1.0]).ndim, 0)

    def test_K_and_K_minus_1_calls_equal(self):
        np.random.seed(2846)
        n = np.random.randint(1, 32)
        alpha = np.random.uniform(1e-09, 100, n)
        d = dirichlet(alpha)
        num_tests = 10
        for i in range(num_tests):
            x = np.random.uniform(1e-09, 100, n)
            x /= np.sum(x)
            assert_almost_equal(d.pdf(x[:-1]), d.pdf(x))

    def test_multiple_entry_calls(self):
        np.random.seed(2846)
        n = np.random.randint(1, 32)
        alpha = np.random.uniform(1e-09, 100, n)
        d = dirichlet(alpha)
        num_tests = 10
        num_multiple = 5
        xm = None
        for i in range(num_tests):
            for m in range(num_multiple):
                x = np.random.uniform(1e-09, 100, n)
                x /= np.sum(x)
                if xm is not None:
                    xm = np.vstack((xm, x))
                else:
                    xm = x
            rm = d.pdf(xm.T)
            rs = None
            for xs in xm:
                r = d.pdf(xs)
                if rs is not None:
                    rs = np.append(rs, r)
                else:
                    rs = r
            assert_array_almost_equal(rm, rs)

    def test_2D_dirichlet_is_beta(self):
        np.random.seed(2846)
        alpha = np.random.uniform(1e-09, 100, 2)
        d = dirichlet(alpha)
        b = beta(alpha[0], alpha[1])
        num_tests = 10
        for i in range(num_tests):
            x = np.random.uniform(1e-09, 100, 2)
            x /= np.sum(x)
            assert_almost_equal(b.pdf(x), d.pdf([x]))
        assert_almost_equal(b.mean(), d.mean()[0])
        assert_almost_equal(b.var(), d.var()[0])