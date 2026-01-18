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
def test_process_parameters(self):
    message = '`row` must be one-dimensional'
    with pytest.raises(ValueError, match=message):
        random_table([[1, 2]], [1, 2])
    message = '`col` must be one-dimensional'
    with pytest.raises(ValueError, match=message):
        random_table([1, 2], [[1, 2]])
    message = 'each element of `row` must be non-negative'
    with pytest.raises(ValueError, match=message):
        random_table([1, -1], [1, 2])
    message = 'each element of `col` must be non-negative'
    with pytest.raises(ValueError, match=message):
        random_table([1, 2], [1, -2])
    message = 'sums over `row` and `col` must be equal'
    with pytest.raises(ValueError, match=message):
        random_table([1, 2], [1, 0])
    message = 'each element of `row` must be an integer'
    with pytest.raises(ValueError, match=message):
        random_table([2.1, 2.1], [1, 1, 2])
    message = 'each element of `col` must be an integer'
    with pytest.raises(ValueError, match=message):
        random_table([1, 2], [1.1, 1.1, 1])
    row = [1, 3]
    col = [2, 1, 1]
    r, c, n = random_table._process_parameters([1, 3], [2, 1, 1])
    assert_equal(row, r)
    assert_equal(col, c)
    assert n == np.sum(row)