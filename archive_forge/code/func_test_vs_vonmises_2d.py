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
def test_vs_vonmises_2d(self):
    rng = np.random.default_rng(2777937887058094419)
    mu = np.array([0, 1])
    mu_angle = np.arctan2(mu[1], mu[0])
    kappa = 20
    vmf = vonmises_fisher(mu, kappa)
    vonmises_dist = vonmises(loc=mu_angle, kappa=kappa)
    vectors = uniform_direction(2).rvs(10, random_state=rng)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    assert_allclose(vonmises_dist.entropy(), vmf.entropy())
    assert_allclose(vonmises_dist.pdf(angles), vmf.pdf(vectors))
    assert_allclose(vonmises_dist.logpdf(angles), vmf.logpdf(vectors))