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
def test_scale_dimensions(self):
    true_scale = np.array(1, ndmin=2)
    scales = [1, [1], np.array(1), np.r_[1], np.array(1, ndmin=2)]
    for scale in scales:
        w = wishart(1, scale)
        assert_equal(w.scale, true_scale)
        assert_equal(w.scale.shape, true_scale.shape)
    true_scale = np.array([[1, 0], [0, 2]])
    scales = [[1, 2], np.r_[1, 2], np.array([[1, 0], [0, 2]])]
    for scale in scales:
        w = wishart(2, scale)
        assert_equal(w.scale, true_scale)
        assert_equal(w.scale.shape, true_scale.shape)
    assert_raises(ValueError, wishart, 1, np.eye(2))
    wishart(1.1, np.eye(2))
    scale = np.array(1, ndmin=3)
    assert_raises(ValueError, wishart, 1, scale)