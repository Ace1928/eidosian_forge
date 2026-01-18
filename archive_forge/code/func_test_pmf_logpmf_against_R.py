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
def test_pmf_logpmf_against_R(self):
    x = np.array([1, 2, 3])
    n = np.sum(x)
    alpha = np.array([3, 4, 5])
    res = dirichlet_multinomial.pmf(x, alpha, n)
    logres = dirichlet_multinomial.logpmf(x, alpha, n)
    ref = 0.08484162895927638
    assert_allclose(res, ref)
    assert_allclose(logres, np.log(ref))
    assert res.shape == logres.shape == ()
    rng = np.random.default_rng(28469824356873456)
    alpha = rng.uniform(0, 100, 10)
    x = rng.integers(0, 10, 10)
    n = np.sum(x, axis=-1)
    res = dirichlet_multinomial(alpha, n).pmf(x)
    logres = dirichlet_multinomial.logpmf(x, alpha, n)
    ref = 3.65409306285992e-16
    assert_allclose(res, ref)
    assert_allclose(logres, np.log(ref))