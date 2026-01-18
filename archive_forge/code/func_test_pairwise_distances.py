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
@pytest.mark.slow
def test_pairwise_distances(self):
    np.random.seed(514)

    def random_ortho(dim):
        u, _s, v = np.linalg.svd(np.random.normal(size=(dim, dim)))
        return np.dot(u, v)
    for dim in range(2, 6):

        def generate_test_statistics(rvs, N=1000, eps=1e-10):
            stats = np.array([np.sum((rvs(dim=dim) - rvs(dim=dim)) ** 2) for _ in range(N)])
            stats += np.random.uniform(-eps, eps, size=stats.shape)
            return stats
        expected = generate_test_statistics(random_ortho)
        actual = generate_test_statistics(scipy.stats.ortho_group.rvs)
        _D, p = scipy.stats.ks_2samp(expected, actual)
        assert_array_less(0.05, p)