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
@pytest.mark.parametrize('dim', [5, 8])
@pytest.mark.parametrize('kappa', [1000000000000000.0, 1e+20, 1e+30])
def test_sampling_high_concentration(self, dim, kappa):
    rng = np.random.default_rng(2777937887058094419)
    mu = np.full((dim,), 1 / np.sqrt(dim))
    vmf_dist = vonmises_fisher(mu, kappa, seed=rng)
    vmf_dist.rvs(10)