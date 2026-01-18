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
def test_ransac_default_residual_threshold():
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(estimator, min_samples=2, random_state=0)
    ransac_estimator.fit(X, y)
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False
    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)