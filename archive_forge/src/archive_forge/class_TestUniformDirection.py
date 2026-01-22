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
class TestUniformDirection:

    @pytest.mark.parametrize('dim', [1, 3])
    @pytest.mark.parametrize('size', [None, 1, 5, (5, 4)])
    def test_samples(self, dim, size):
        rng = np.random.default_rng(2777937887058094419)
        uniform_direction_dist = uniform_direction(dim, seed=rng)
        samples = uniform_direction_dist.rvs(size)
        mean, cov = (np.zeros(dim), np.eye(dim))
        expected_shape = rng.multivariate_normal(mean, cov, size=size).shape
        assert samples.shape == expected_shape
        norms = np.linalg.norm(samples, axis=-1)
        assert_allclose(norms, 1.0)

    @pytest.mark.parametrize('dim', [None, 0, (2, 2), 2.5])
    def test_invalid_dim(self, dim):
        message = 'Dimension of vector must be specified, and must be an integer greater than 0.'
        with pytest.raises(ValueError, match=message):
            uniform_direction.rvs(dim)

    def test_frozen_distribution(self):
        dim = 5
        frozen = uniform_direction(dim)
        frozen_seed = uniform_direction(dim, seed=514)
        rvs1 = frozen.rvs(random_state=514)
        rvs2 = uniform_direction.rvs(dim, random_state=514)
        rvs3 = frozen_seed.rvs()
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    @pytest.mark.parametrize('dim', [2, 5, 8])
    def test_uniform(self, dim):
        rng = np.random.default_rng(1036978481269651776)
        spherical_dist = uniform_direction(dim, seed=rng)
        v1, v2 = spherical_dist.rvs(size=2)
        v2 -= v1 @ v2 * v1
        v2 /= np.linalg.norm(v2)
        assert_allclose(v1 @ v2, 0, atol=1e-14)
        samples = spherical_dist.rvs(size=10000)
        s1 = samples @ v1
        s2 = samples @ v2
        angles = np.arctan2(s1, s2)
        angles += np.pi
        angles /= 2 * np.pi
        uniform_dist = uniform()
        kstest_result = kstest(angles, uniform_dist.cdf)
        assert kstest_result.pvalue > 0.05