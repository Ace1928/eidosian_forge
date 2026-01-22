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
class TestMultivariateNormal:

    def test_input_shape(self):
        mu = np.arange(3)
        cov = np.identity(2)
        assert_raises(ValueError, multivariate_normal.pdf, (0, 1), mu, cov)
        assert_raises(ValueError, multivariate_normal.pdf, (0, 1, 2), mu, cov)
        assert_raises(ValueError, multivariate_normal.cdf, (0, 1), mu, cov)
        assert_raises(ValueError, multivariate_normal.cdf, (0, 1, 2), mu, cov)

    def test_scalar_values(self):
        np.random.seed(1234)
        x, mean, cov = (1.5, 1.7, 2.5)
        pdf = multivariate_normal.pdf(x, mean, cov)
        assert_equal(pdf.ndim, 0)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        pdf = multivariate_normal.pdf(x, mean, cov)
        assert_equal(pdf.ndim, 0)
        x, mean, cov = (1.5, 1.7, 2.5)
        cdf = multivariate_normal.cdf(x, mean, cov)
        assert_equal(cdf.ndim, 0)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        cdf = multivariate_normal.cdf(x, mean, cov)
        assert_equal(cdf.ndim, 0)

    def test_logpdf(self):
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        d1 = multivariate_normal.logpdf(x, mean, cov)
        d2 = multivariate_normal.pdf(x, mean, cov)
        assert_allclose(d1, np.log(d2))

    def test_logpdf_default_values(self):
        np.random.seed(1234)
        x = np.random.randn(5)
        d1 = multivariate_normal.logpdf(x)
        d2 = multivariate_normal.pdf(x)
        d3 = multivariate_normal.logpdf(x, None, 1)
        d4 = multivariate_normal.pdf(x, None, 1)
        assert_allclose(d1, np.log(d2))
        assert_allclose(d3, np.log(d4))

    def test_logcdf(self):
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        d1 = multivariate_normal.logcdf(x, mean, cov)
        d2 = multivariate_normal.cdf(x, mean, cov)
        assert_allclose(d1, np.log(d2))

    def test_logcdf_default_values(self):
        np.random.seed(1234)
        x = np.random.randn(5)
        d1 = multivariate_normal.logcdf(x)
        d2 = multivariate_normal.cdf(x)
        d3 = multivariate_normal.logcdf(x, None, 1)
        d4 = multivariate_normal.cdf(x, None, 1)
        assert_allclose(d1, np.log(d2))
        assert_allclose(d3, np.log(d4))

    def test_rank(self):
        np.random.seed(1234)
        n = 4
        mean = np.random.randn(n)
        for expected_rank in range(1, n + 1):
            s = np.random.randn(n, expected_rank)
            cov = np.dot(s, s.T)
            distn = multivariate_normal(mean, cov, allow_singular=True)
            assert_equal(distn.cov_object.rank, expected_rank)

    def test_degenerate_distributions(self):
        for n in range(1, 5):
            z = np.random.randn(n)
            for k in range(1, n):
                s = np.random.randn(k, k)
                cov_kk = np.dot(s, s.T)
                cov_nn = np.zeros((n, n))
                cov_nn[:k, :k] = cov_kk
                x = np.zeros(n)
                x[:k] = z[:k]
                u = _sample_orthonormal_matrix(n)
                cov_rr = np.dot(u, np.dot(cov_nn, u.T))
                y = np.dot(u, x)
                distn_kk = multivariate_normal(np.zeros(k), cov_kk, allow_singular=True)
                distn_nn = multivariate_normal(np.zeros(n), cov_nn, allow_singular=True)
                distn_rr = multivariate_normal(np.zeros(n), cov_rr, allow_singular=True)
                assert_equal(distn_kk.cov_object.rank, k)
                assert_equal(distn_nn.cov_object.rank, k)
                assert_equal(distn_rr.cov_object.rank, k)
                pdf_kk = distn_kk.pdf(x[:k])
                pdf_nn = distn_nn.pdf(x)
                pdf_rr = distn_rr.pdf(y)
                assert_allclose(pdf_kk, pdf_nn)
                assert_allclose(pdf_kk, pdf_rr)
                logpdf_kk = distn_kk.logpdf(x[:k])
                logpdf_nn = distn_nn.logpdf(x)
                logpdf_rr = distn_rr.logpdf(y)
                assert_allclose(logpdf_kk, logpdf_nn)
                assert_allclose(logpdf_kk, logpdf_rr)
                y_orth = y + u[:, -1]
                pdf_rr_orth = distn_rr.pdf(y_orth)
                logpdf_rr_orth = distn_rr.logpdf(y_orth)
                assert_equal(pdf_rr_orth, 0.0)
                assert_equal(logpdf_rr_orth, -np.inf)

    def test_degenerate_array(self):
        k = 10
        for n in range(2, 6):
            for r in range(1, n):
                mn = np.zeros(n)
                u = _sample_orthonormal_matrix(n)[:, :r]
                vr = np.dot(u, u.T)
                X = multivariate_normal.rvs(mean=mn, cov=vr, size=k)
                pdf = multivariate_normal.pdf(X, mean=mn, cov=vr, allow_singular=True)
                assert_equal(pdf.size, k)
                assert np.all(pdf > 0.0)
                logpdf = multivariate_normal.logpdf(X, mean=mn, cov=vr, allow_singular=True)
                assert_equal(logpdf.size, k)
                assert np.all(logpdf > -np.inf)

    def test_large_pseudo_determinant(self):
        large_total_log = 1000.0
        npos = 100
        nzero = 2
        large_entry = np.exp(large_total_log / npos)
        n = npos + nzero
        cov = np.zeros((n, n), dtype=float)
        np.fill_diagonal(cov, large_entry)
        cov[-nzero:, -nzero:] = 0
        assert_equal(scipy.linalg.det(cov), 0)
        assert_equal(scipy.linalg.det(cov[:npos, :npos]), np.inf)
        assert_allclose(np.linalg.slogdet(cov[:npos, :npos]), (1, large_total_log))
        psd = _PSD(cov)
        assert_allclose(psd.log_pdet, large_total_log)

    def test_broadcasting(self):
        np.random.seed(1234)
        n = 4
        data = np.random.randn(n, n)
        cov = np.dot(data, data.T)
        mean = np.random.randn(n)
        X = np.random.randn(2, 3, n)
        desired_pdf = multivariate_normal.pdf(X, mean, cov)
        desired_cdf = multivariate_normal.cdf(X, mean, cov)
        for i in range(2):
            for j in range(3):
                actual = multivariate_normal.pdf(X[i, j], mean, cov)
                assert_allclose(actual, desired_pdf[i, j])
                actual = multivariate_normal.cdf(X[i, j], mean, cov)
                assert_allclose(actual, desired_cdf[i, j], rtol=0.001)

    def test_normal_1D(self):
        x = np.linspace(0, 2, 10)
        mean, cov = (1.2, 0.9)
        scale = cov ** 0.5
        d1 = norm.pdf(x, mean, scale)
        d2 = multivariate_normal.pdf(x, mean, cov)
        assert_allclose(d1, d2)
        d1 = norm.cdf(x, mean, scale)
        d2 = multivariate_normal.cdf(x, mean, cov)
        assert_allclose(d1, d2)

    def test_marginalization(self):
        mean = np.array([2.5, 3.5])
        cov = np.array([[0.5, 0.2], [0.2, 0.6]])
        n = 2 ** 8 + 1
        delta = 6 / (n - 1)
        v = np.linspace(0, 6, n)
        xv, yv = np.meshgrid(v, v)
        pos = np.empty((n, n, 2))
        pos[:, :, 0] = xv
        pos[:, :, 1] = yv
        pdf = multivariate_normal.pdf(pos, mean, cov)
        margin_x = romb(pdf, delta, axis=0)
        margin_y = romb(pdf, delta, axis=1)
        gauss_x = norm.pdf(v, loc=mean[0], scale=cov[0, 0] ** 0.5)
        gauss_y = norm.pdf(v, loc=mean[1], scale=cov[1, 1] ** 0.5)
        assert_allclose(margin_x, gauss_x, rtol=0.01, atol=0.01)
        assert_allclose(margin_y, gauss_y, rtol=0.01, atol=0.01)

    def test_frozen(self):
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        norm_frozen = multivariate_normal(mean, cov)
        assert_allclose(norm_frozen.pdf(x), multivariate_normal.pdf(x, mean, cov))
        assert_allclose(norm_frozen.logpdf(x), multivariate_normal.logpdf(x, mean, cov))
        assert_allclose(norm_frozen.cdf(x), multivariate_normal.cdf(x, mean, cov))
        assert_allclose(norm_frozen.logcdf(x), multivariate_normal.logcdf(x, mean, cov))

    @pytest.mark.parametrize('covariance', [np.eye(2), Covariance.from_diagonal([1, 1])])
    def test_frozen_multivariate_normal_exposes_attributes(self, covariance):
        mean = np.ones((2,))
        cov_should_be = np.eye(2)
        norm_frozen = multivariate_normal(mean, covariance)
        assert np.allclose(norm_frozen.mean, mean)
        assert np.allclose(norm_frozen.cov, cov_should_be)

    def test_pseudodet_pinv(self):
        np.random.seed(1234)
        n = 7
        x = np.random.randn(n, n)
        cov = np.dot(x, x.T)
        s, u = scipy.linalg.eigh(cov)
        s = np.full(n, 0.5)
        s[0] = 1.0
        s[-1] = 1e-07
        cov = np.dot(u, np.dot(np.diag(s), u.T))
        cond = 1e-05
        psd = _PSD(cov, cond=cond)
        psd_pinv = _PSD(psd.pinv, cond=cond)
        assert_allclose(psd.log_pdet, np.sum(np.log(s[:-1])))
        assert_allclose(-psd.log_pdet, psd_pinv.log_pdet)

    def test_exception_nonsquare_cov(self):
        cov = [[1, 2, 3], [4, 5, 6]]
        assert_raises(ValueError, _PSD, cov)

    def test_exception_nonfinite_cov(self):
        cov_nan = [[1, 0], [0, np.nan]]
        assert_raises(ValueError, _PSD, cov_nan)
        cov_inf = [[1, 0], [0, np.inf]]
        assert_raises(ValueError, _PSD, cov_inf)

    def test_exception_non_psd_cov(self):
        cov = [[1, 0], [0, -1]]
        assert_raises(ValueError, _PSD, cov)

    def test_exception_singular_cov(self):
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.ones((5, 5))
        e = np.linalg.LinAlgError
        assert_raises(e, multivariate_normal, mean, cov)
        assert_raises(e, multivariate_normal.pdf, x, mean, cov)
        assert_raises(e, multivariate_normal.logpdf, x, mean, cov)
        assert_raises(e, multivariate_normal.cdf, x, mean, cov)
        assert_raises(e, multivariate_normal.logcdf, x, mean, cov)
        cov = [[1.0, 0.0], [1.0, 1.0]]
        msg = 'When `allow_singular is False`, the input matrix'
        with pytest.raises(np.linalg.LinAlgError, match=msg):
            multivariate_normal(cov=cov)

    def test_R_values(self):
        r_pdf = np.array([0.0002214706, 0.0013819953, 0.0049138692, 0.010380305, 0.01402508])
        x = np.linspace(0, 2, 5)
        y = 3 * x - 2
        z = x + np.cos(y)
        r = np.array([x, y, z]).T
        mean = np.array([1, 3, 2], 'd')
        cov = np.array([[1, 2, 0], [2, 5, 0.5], [0, 0.5, 3]], 'd')
        pdf = multivariate_normal.pdf(r, mean, cov)
        assert_allclose(pdf, r_pdf, atol=1e-10)
        r_cdf = np.array([0.0017866215, 0.0267142892, 0.0857098761, 0.1063242573, 0.2501068509])
        cdf = multivariate_normal.cdf(r, mean, cov)
        assert_allclose(cdf, r_cdf, atol=2e-05)
        r_cdf2 = np.array([0.01262147, 0.05838989, 0.18389571, 0.40696599, 0.66470577])
        r2 = np.array([x, y]).T
        mean2 = np.array([1, 3], 'd')
        cov2 = np.array([[1, 2], [2, 5]], 'd')
        cdf2 = multivariate_normal.cdf(r2, mean2, cov2)
        assert_allclose(cdf2, r_cdf2, atol=1e-05)

    def test_multivariate_normal_rvs_zero_covariance(self):
        mean = np.zeros(2)
        covariance = np.zeros((2, 2))
        model = multivariate_normal(mean, covariance, allow_singular=True)
        sample = model.rvs()
        assert_equal(sample, [0, 0])

    def test_rvs_shape(self):
        N = 300
        d = 4
        sample = multivariate_normal.rvs(mean=np.zeros(d), cov=1, size=N)
        assert_equal(sample.shape, (N, d))
        sample = multivariate_normal.rvs(mean=None, cov=np.array([[2, 0.1], [0.1, 1]]), size=N)
        assert_equal(sample.shape, (N, 2))
        u = multivariate_normal(mean=0, cov=1)
        sample = u.rvs(N)
        assert_equal(sample.shape, (N,))

    def test_large_sample(self):
        np.random.seed(2846)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        size = 5000
        sample = multivariate_normal.rvs(mean, cov, size)
        assert_allclose(numpy.cov(sample.T), cov, rtol=0.1)
        assert_allclose(sample.mean(0), mean, rtol=0.1)

    def test_entropy(self):
        np.random.seed(2846)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        rv = multivariate_normal(mean, cov)
        assert_almost_equal(rv.entropy(), multivariate_normal.entropy(mean, cov))
        eigs = np.linalg.eig(cov)[0]
        desired = 1 / 2 * (n * (np.log(2 * np.pi) + 1) + np.sum(np.log(eigs)))
        assert_almost_equal(desired, rv.entropy())

    def test_lnB(self):
        alpha = np.array([1, 1, 1])
        desired = 0.5
        assert_almost_equal(np.exp(_lnB(alpha)), desired)

    def test_cdf_with_lower_limit_arrays(self):
        rng = np.random.default_rng(2408071309372769818)
        mean = [0, 0]
        cov = np.eye(2)
        a = rng.random((4, 3, 2)) * 6 - 3
        b = rng.random((4, 3, 2)) * 6 - 3
        cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        cdf2a = multivariate_normal.cdf(b, mean, cov)
        cdf2b = multivariate_normal.cdf(a, mean, cov)
        ab1 = np.concatenate((a[..., 0:1], b[..., 1:2]), axis=-1)
        ab2 = np.concatenate((a[..., 1:2], b[..., 0:1]), axis=-1)
        cdf2ab1 = multivariate_normal.cdf(ab1, mean, cov)
        cdf2ab2 = multivariate_normal.cdf(ab2, mean, cov)
        cdf2 = cdf2a + cdf2b - cdf2ab1 - cdf2ab2
        assert_allclose(cdf1, cdf2)

    def test_cdf_with_lower_limit_consistency(self):
        rng = np.random.default_rng(2408071309372769818)
        mean = rng.random(3)
        cov = rng.random((3, 3))
        cov = cov @ cov.T
        a = rng.random((2, 3)) * 6 - 3
        b = rng.random((2, 3)) * 6 - 3
        cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        cdf2 = multivariate_normal(mean, cov).cdf(b, lower_limit=a)
        cdf3 = np.exp(multivariate_normal.logcdf(b, mean, cov, lower_limit=a))
        cdf4 = np.exp(multivariate_normal(mean, cov).logcdf(b, lower_limit=a))
        assert_allclose(cdf2, cdf1, rtol=0.0001)
        assert_allclose(cdf3, cdf1, rtol=0.0001)
        assert_allclose(cdf4, cdf1, rtol=0.0001)

    def test_cdf_signs(self):
        mean = np.zeros(3)
        cov = np.eye(3)
        b = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]
        a = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
        expected_signs = np.array([1, -1, -1, 1])
        cdf = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        assert_allclose(cdf, cdf[0] * expected_signs)

    def test_mean_cov(self):
        P = np.diag(1 / np.array([1, 2, 3]))
        cov_object = _covariance.CovViaPrecision(P)
        message = '`cov` represents a covariance matrix in 3 dimensions...'
        with pytest.raises(ValueError, match=message):
            multivariate_normal.entropy([0, 0], cov_object)
        with pytest.raises(ValueError, match=message):
            multivariate_normal([0, 0], cov_object)
        x = [0.5, 0.5, 0.5]
        ref = multivariate_normal.pdf(x, [0, 0, 0], cov_object)
        assert_equal(multivariate_normal.pdf(x, cov=cov_object), ref)
        ref = multivariate_normal.pdf(x, [1, 1, 1], cov_object)
        assert_equal(multivariate_normal.pdf(x, 1, cov=cov_object), ref)

    def test_fit_wrong_fit_data_shape(self):
        data = [1, 3]
        error_msg = '`x` must be two-dimensional.'
        with pytest.raises(ValueError, match=error_msg):
            multivariate_normal.fit(data)

    @pytest.mark.parametrize('dim', (3, 5))
    def test_fit_correctness(self, dim):
        rng = np.random.default_rng(4385269356937404)
        x = rng.random((100, dim))
        mean_est, cov_est = multivariate_normal.fit(x)
        mean_ref, cov_ref = (np.mean(x, axis=0), np.cov(x.T, ddof=0))
        assert_allclose(mean_est, mean_ref, atol=1e-15)
        assert_allclose(cov_est, cov_ref, rtol=1e-15)

    def test_fit_both_parameters_fixed(self):
        data = np.full((2, 1), 3)
        mean_fixed = 1.0
        cov_fixed = np.atleast_2d(1.0)
        mean, cov = multivariate_normal.fit(data, fix_mean=mean_fixed, fix_cov=cov_fixed)
        assert_equal(mean, mean_fixed)
        assert_equal(cov, cov_fixed)

    @pytest.mark.parametrize('fix_mean', [np.zeros((2, 2)), np.zeros((3,))])
    def test_fit_fix_mean_input_validation(self, fix_mean):
        msg = '`fix_mean` must be a one-dimensional array the same length as the dimensionality of the vectors `x`.'
        with pytest.raises(ValueError, match=msg):
            multivariate_normal.fit(np.eye(2), fix_mean=fix_mean)

    @pytest.mark.parametrize('fix_cov', [np.zeros((2,)), np.zeros((3, 2)), np.zeros((4, 4))])
    def test_fit_fix_cov_input_validation_dimension(self, fix_cov):
        msg = '`fix_cov` must be a two-dimensional square array of same side length as the dimensionality of the vectors `x`.'
        with pytest.raises(ValueError, match=msg):
            multivariate_normal.fit(np.eye(3), fix_cov=fix_cov)

    def test_fit_fix_cov_not_positive_semidefinite(self):
        error_msg = '`fix_cov` must be symmetric positive semidefinite.'
        with pytest.raises(ValueError, match=error_msg):
            fix_cov = np.array([[1.0, 0.0], [0.0, -1.0]])
            multivariate_normal.fit(np.eye(2), fix_cov=fix_cov)

    def test_fit_fix_mean(self):
        rng = np.random.default_rng(4385269356937404)
        loc = rng.random(3)
        A = rng.random((3, 3))
        cov = np.dot(A, A.T)
        samples = multivariate_normal.rvs(mean=loc, cov=cov, size=100, random_state=rng)
        mean_free, cov_free = multivariate_normal.fit(samples)
        logp_free = multivariate_normal.logpdf(samples, mean=mean_free, cov=cov_free).sum()
        mean_fix, cov_fix = multivariate_normal.fit(samples, fix_mean=loc)
        assert_equal(mean_fix, loc)
        logp_fix = multivariate_normal.logpdf(samples, mean=mean_fix, cov=cov_fix).sum()
        assert logp_fix < logp_free
        A = rng.random((3, 3))
        m = 1e-08 * np.dot(A, A.T)
        cov_perturbed = cov_fix + m
        logp_perturbed = multivariate_normal.logpdf(samples, mean=mean_fix, cov=cov_perturbed).sum()
        assert logp_perturbed < logp_fix

    def test_fit_fix_cov(self):
        rng = np.random.default_rng(4385269356937404)
        loc = rng.random(3)
        A = rng.random((3, 3))
        cov = np.dot(A, A.T)
        samples = multivariate_normal.rvs(mean=loc, cov=cov, size=100, random_state=rng)
        mean_free, cov_free = multivariate_normal.fit(samples)
        logp_free = multivariate_normal.logpdf(samples, mean=mean_free, cov=cov_free).sum()
        mean_fix, cov_fix = multivariate_normal.fit(samples, fix_cov=cov)
        assert_equal(mean_fix, np.mean(samples, axis=0))
        assert_equal(cov_fix, cov)
        logp_fix = multivariate_normal.logpdf(samples, mean=mean_fix, cov=cov_fix).sum()
        assert logp_fix < logp_free
        mean_perturbed = mean_fix + 1e-08 * rng.random(3)
        logp_perturbed = multivariate_normal.logpdf(samples, mean=mean_perturbed, cov=cov_fix).sum()
        assert logp_perturbed < logp_fix