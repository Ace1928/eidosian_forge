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
@pytest.mark.parametrize('x, mu, kappa, reference', [(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0001, 0.0795854295583605), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 0.0001, 0.07957747141331854), (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 15.915494309189533), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 100, 5.920684802611232e-43), (np.array([1.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0]), 2000, 5.930499050746588e-07), (np.array([1.0, 0.0, 0]), np.array([1.0, 0.0, 0.0]), 2000, 318.3098861837907), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 2000, 101371.86957712633), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0, 0, 0.0]), 2000, 0.00018886808182653578), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.8), np.sqrt(0.2), 0.0, 0, 0.0]), 2000, 2.0255393314603194e-87)])
def test_pdf_accuracy(self, x, mu, kappa, reference):
    pdf = vonmises_fisher(mu, kappa).pdf(x)
    assert_allclose(pdf, reference, rtol=1e-13)