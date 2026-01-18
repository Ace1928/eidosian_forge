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
def test_ransac_stop_n_inliers():
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(estimator, min_samples=2, residual_threshold=5, stop_n_inliers=2, random_state=0)
    ransac_estimator.fit(X, y)
    assert ransac_estimator.n_trials_ == 1