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
@pytest.mark.parametrize('method', ('boyett', 'patefield'))
def test_rvs_size(self, method):
    row = [2, 6]
    col = [1, 3, 4]
    rv = random_table.rvs(row, col, method=method, random_state=self.get_rng())
    assert rv.shape == (2, 3)
    rv2 = random_table.rvs(row, col, size=1, method=method, random_state=self.get_rng())
    assert rv2.shape == (1, 2, 3)
    assert_equal(rv, rv2[0])
    rv3 = random_table.rvs(row, col, size=0, method=method, random_state=self.get_rng())
    assert rv3.shape == (0, 2, 3)
    rv4 = random_table.rvs(row, col, size=20, method=method, random_state=self.get_rng())
    assert rv4.shape == (20, 2, 3)
    rv5 = random_table.rvs(row, col, size=(4, 5), method=method, random_state=self.get_rng())
    assert rv5.shape == (4, 5, 2, 3)
    assert_allclose(rv5.reshape(20, 2, 3), rv4, rtol=1e-15)
    message = '`size` must be a non-negative integer or `None`'
    with pytest.raises(ValueError, match=message):
        random_table.rvs(row, col, size=-1, method=method, random_state=self.get_rng())
    with pytest.raises(ValueError, match=message):
        random_table.rvs(row, col, size=np.nan, method=method, random_state=self.get_rng())