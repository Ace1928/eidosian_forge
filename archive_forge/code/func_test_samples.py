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
@pytest.mark.parametrize('dim', [2, 3, 4, 6])
@pytest.mark.parametrize('size', [None, 1, 5, (5, 4)])
def test_samples(self, dim, size):
    rng = np.random.default_rng(2777937887058094419)
    mu = np.full((dim,), 1 / np.sqrt(dim))
    vmf_dist = vonmises_fisher(mu, 1, seed=rng)
    samples = vmf_dist.rvs(size)
    mean, cov = (np.zeros(dim), np.eye(dim))
    expected_shape = rng.multivariate_normal(mean, cov, size=size).shape
    assert samples.shape == expected_shape
    norms = np.linalg.norm(samples, axis=-1)
    assert_allclose(norms, 1.0)