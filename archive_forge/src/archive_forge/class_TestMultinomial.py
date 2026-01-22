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
class TestMultinomial:

    def test_logpmf(self):
        vals1 = multinomial.logpmf((3, 4), 7, (0.3, 0.7))
        assert_allclose(vals1, -1.483270127243324, rtol=1e-08)
        vals2 = multinomial.logpmf([3, 4], 0, [0.3, 0.7])
        assert vals2 == -np.inf
        vals3 = multinomial.logpmf([0, 0], 0, [0.3, 0.7])
        assert vals3 == 0
        vals4 = multinomial.logpmf([3, 4], 0, [-2, 3])
        assert_allclose(vals4, np.nan, rtol=1e-08)

    def test_reduces_binomial(self):
        val1 = multinomial.logpmf((3, 4), 7, (0.3, 0.7))
        val2 = binom.logpmf(3, 7, 0.3)
        assert_allclose(val1, val2, rtol=1e-08)
        val1 = multinomial.pmf((6, 8), 14, (0.1, 0.9))
        val2 = binom.pmf(6, 14, 0.1)
        assert_allclose(val1, val2, rtol=1e-08)

    def test_R(self):
        n, p = (3, [1.0 / 8, 2.0 / 8, 5.0 / 8])
        r_vals = {(0, 0, 3): 0.244140625, (1, 0, 2): 0.146484375, (2, 0, 1): 0.029296875, (3, 0, 0): 0.001953125, (0, 1, 2): 0.29296875, (1, 1, 1): 0.1171875, (2, 1, 0): 0.01171875, (0, 2, 1): 0.1171875, (1, 2, 0): 0.0234375, (0, 3, 0): 0.015625}
        for x in r_vals:
            assert_allclose(multinomial.pmf(x, n, p), r_vals[x], atol=1e-14)

    @pytest.mark.parametrize('n', [0, 3])
    def test_rvs_np(self, n):
        sc_rvs = multinomial.rvs(n, [1 / 4.0] * 3, size=7, random_state=123)
        rndm = np.random.RandomState(123)
        np_rvs = rndm.multinomial(n, [1 / 4.0] * 3, size=7)
        assert_equal(sc_rvs, np_rvs)

    def test_pmf(self):
        vals0 = multinomial.pmf((5,), 5, (1,))
        assert_allclose(vals0, 1, rtol=1e-08)
        vals1 = multinomial.pmf((3, 4), 7, (0.3, 0.7))
        assert_allclose(vals1, 0.22689449999999994, rtol=1e-08)
        vals2 = multinomial.pmf([[[3, 5], [0, 8]], [[-1, 9], [1, 1]]], 8, (0.1, 0.9))
        assert_allclose(vals2, [[0.03306744, 0.43046721], [0, 0]], rtol=1e-08)
        x = np.empty((0, 2), dtype=np.float64)
        vals3 = multinomial.pmf(x, 4, (0.3, 0.7))
        assert_equal(vals3, np.empty([], dtype=np.float64))
        vals4 = multinomial.pmf([1, 2], 4, (0.3, 0.7))
        assert_allclose(vals4, 0, rtol=1e-08)
        vals5 = multinomial.pmf([3, 3, 0], 6, [2 / 3.0, 1 / 3.0, 0])
        assert_allclose(vals5, 0.219478737997, rtol=1e-08)
        vals5 = multinomial.pmf([0, 0, 0], 0, [2 / 3.0, 1 / 3.0, 0])
        assert vals5 == 1
        vals6 = multinomial.pmf([2, 1, 0], 0, [2 / 3.0, 1 / 3.0, 0])
        assert vals6 == 0

    def test_pmf_broadcasting(self):
        vals0 = multinomial.pmf([1, 2], 3, [[0.1, 0.9], [0.2, 0.8]])
        assert_allclose(vals0, [0.243, 0.384], rtol=1e-08)
        vals1 = multinomial.pmf([1, 2], [3, 4], [0.1, 0.9])
        assert_allclose(vals1, [0.243, 0], rtol=1e-08)
        vals2 = multinomial.pmf([[[1, 2], [1, 1]]], 3, [0.1, 0.9])
        assert_allclose(vals2, [[0.243, 0]], rtol=1e-08)
        vals3 = multinomial.pmf([1, 2], [[[3], [4]]], [0.1, 0.9])
        assert_allclose(vals3, [[[0.243], [0]]], rtol=1e-08)
        vals4 = multinomial.pmf([[1, 2], [1, 1]], [[[[3]]]], [0.1, 0.9])
        assert_allclose(vals4, [[[[0.243, 0]]]], rtol=1e-08)

    @pytest.mark.parametrize('n', [0, 5])
    def test_cov(self, n):
        cov1 = multinomial.cov(n, (0.2, 0.3, 0.5))
        cov2 = [[n * 0.2 * 0.8, -n * 0.2 * 0.3, -n * 0.2 * 0.5], [-n * 0.3 * 0.2, n * 0.3 * 0.7, -n * 0.3 * 0.5], [-n * 0.5 * 0.2, -n * 0.5 * 0.3, n * 0.5 * 0.5]]
        assert_allclose(cov1, cov2, rtol=1e-08)

    def test_cov_broadcasting(self):
        cov1 = multinomial.cov(5, [[0.1, 0.9], [0.2, 0.8]])
        cov2 = [[[0.45, -0.45], [-0.45, 0.45]], [[0.8, -0.8], [-0.8, 0.8]]]
        assert_allclose(cov1, cov2, rtol=1e-08)
        cov3 = multinomial.cov([4, 5], [0.1, 0.9])
        cov4 = [[[0.36, -0.36], [-0.36, 0.36]], [[0.45, -0.45], [-0.45, 0.45]]]
        assert_allclose(cov3, cov4, rtol=1e-08)
        cov5 = multinomial.cov([4, 5], [[0.3, 0.7], [0.4, 0.6]])
        cov6 = [[[4 * 0.3 * 0.7, -4 * 0.3 * 0.7], [-4 * 0.3 * 0.7, 4 * 0.3 * 0.7]], [[5 * 0.4 * 0.6, -5 * 0.4 * 0.6], [-5 * 0.4 * 0.6, 5 * 0.4 * 0.6]]]
        assert_allclose(cov5, cov6, rtol=1e-08)

    @pytest.mark.parametrize('n', [0, 2])
    def test_entropy(self, n):
        ent0 = multinomial.entropy(n, [0.2, 0.8])
        assert_allclose(ent0, binom.entropy(n, 0.2), rtol=1e-08)

    def test_entropy_broadcasting(self):
        ent0 = multinomial.entropy([2, 3], [0.2, 0.3])
        assert_allclose(ent0, [binom.entropy(2, 0.2), binom.entropy(3, 0.2)], rtol=1e-08)
        ent1 = multinomial.entropy([7, 8], [[0.3, 0.7], [0.4, 0.6]])
        assert_allclose(ent1, [binom.entropy(7, 0.3), binom.entropy(8, 0.4)], rtol=1e-08)
        ent2 = multinomial.entropy([[7], [8]], [[0.3, 0.7], [0.4, 0.6]])
        assert_allclose(ent2, [[binom.entropy(7, 0.3), binom.entropy(7, 0.4)], [binom.entropy(8, 0.3), binom.entropy(8, 0.4)]], rtol=1e-08)

    @pytest.mark.parametrize('n', [0, 5])
    def test_mean(self, n):
        mean1 = multinomial.mean(n, [0.2, 0.8])
        assert_allclose(mean1, [n * 0.2, n * 0.8], rtol=1e-08)

    def test_mean_broadcasting(self):
        mean1 = multinomial.mean([5, 6], [0.2, 0.8])
        assert_allclose(mean1, [[5 * 0.2, 5 * 0.8], [6 * 0.2, 6 * 0.8]], rtol=1e-08)

    def test_frozen(self):
        np.random.seed(1234)
        n = 12
        pvals = (0.1, 0.2, 0.3, 0.4)
        x = [[0, 0, 0, 12], [0, 0, 1, 11], [0, 1, 1, 10], [1, 1, 1, 9], [1, 1, 2, 8]]
        x = np.asarray(x, dtype=np.float64)
        mn_frozen = multinomial(n, pvals)
        assert_allclose(mn_frozen.pmf(x), multinomial.pmf(x, n, pvals))
        assert_allclose(mn_frozen.logpmf(x), multinomial.logpmf(x, n, pvals))
        assert_allclose(mn_frozen.entropy(), multinomial.entropy(n, pvals))

    def test_gh_11860(self):
        n = 88
        rng = np.random.default_rng(8879715917488330089)
        p = rng.random(n)
        p[-1] = 1e-30
        p /= np.sum(p)
        x = np.ones(n)
        logpmf = multinomial.logpmf(x, n, p)
        assert np.isfinite(logpmf)