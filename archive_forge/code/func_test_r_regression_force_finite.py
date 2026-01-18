import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('X, y, expected_corr_coef, force_finite', [(np.array([[2, 1], [2, 0], [2, 10], [2, 4]]), np.array([0, 1, 1, 0]), np.array([0.0, 0.32075]), True), (np.array([[5, 1], [3, 0], [2, 10], [8, 4]]), np.array([0, 0, 0, 0]), np.array([0.0, 0.0]), True), (np.array([[2, 1], [2, 0], [2, 10], [2, 4]]), np.array([0, 1, 1, 0]), np.array([np.nan, 0.32075]), False), (np.array([[5, 1], [3, 0], [2, 10], [8, 4]]), np.array([0, 0, 0, 0]), np.array([np.nan, np.nan]), False)])
def test_r_regression_force_finite(X, y, expected_corr_coef, force_finite):
    """Check the behaviour of `force_finite` for some corner cases with `r_regression`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/15672
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        corr_coef = r_regression(X, y, force_finite=force_finite)
    np.testing.assert_array_almost_equal(corr_coef, expected_corr_coef)