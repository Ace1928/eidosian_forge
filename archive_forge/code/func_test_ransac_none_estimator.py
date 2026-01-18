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
def test_ransac_none_estimator():
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(estimator, min_samples=2, residual_threshold=5, random_state=0)
    ransac_none_estimator = RANSACRegressor(None, min_samples=2, residual_threshold=5, random_state=0)
    ransac_estimator.fit(X, y)
    ransac_none_estimator.fit(X, y)
    assert_array_almost_equal(ransac_estimator.predict(X), ransac_none_estimator.predict(X))