import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ransac import _dynamic_max_trials
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_perfect_horizontal_line():
    """Check that we can fit a line where all samples are inliers.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19497
    """
    X = np.arange(100)[:, None]
    y = np.zeros((100,))
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(estimator, random_state=0)
    ransac_estimator.fit(X, y)
    assert_allclose(ransac_estimator.estimator_.coef_, 0.0)
    assert_allclose(ransac_estimator.estimator_.intercept_, 0.0)