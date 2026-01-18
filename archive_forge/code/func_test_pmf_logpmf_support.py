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
def test_pmf_logpmf_support(self):
    rng, m, alpha, n, x = self.get_params(1)
    n += 1
    assert_equal(dirichlet_multinomial(alpha, n).pmf(x), 0)
    assert_equal(dirichlet_multinomial(alpha, n).logpmf(x), -np.inf)
    rng, m, alpha, n, x = self.get_params(10)
    i = rng.random(size=10) > 0.5
    x[i] = np.round(x[i] * 2)
    assert_equal(dirichlet_multinomial(alpha, n).pmf(x)[i], 0)
    assert_equal(dirichlet_multinomial(alpha, n).logpmf(x)[i], -np.inf)
    assert np.all(dirichlet_multinomial(alpha, n).pmf(x)[~i] > 0)
    assert np.all(dirichlet_multinomial(alpha, n).logpmf(x)[~i] > -np.inf)