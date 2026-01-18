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
def test_sample_mean(self):
    """Test that sample mean consistent with known mean."""
    df = 10
    sample_size = 20000
    for dim in [1, 5]:
        scale = np.diag(np.arange(dim) + 1)
        scale[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) / 2)
        scale = np.dot(scale.T, scale)
        dist = invwishart(df, scale)
        Xmean_exp = dist.mean()
        Xvar_exp = dist.var()
        Xmean_std = (Xvar_exp / sample_size) ** 0.5
        X = dist.rvs(size=sample_size, random_state=1234)
        Xmean_est = X.mean(axis=0)
        ntests = dim * (dim + 1) // 2
        fail_rate = 0.01 / ntests
        max_diff = norm.ppf(1 - fail_rate / 2)
        assert np.allclose((Xmean_est - Xmean_exp) / Xmean_std, 0, atol=max_diff)