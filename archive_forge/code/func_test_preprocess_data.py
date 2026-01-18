import warnings
import numpy as np
import pytest
from scipy import linalg, sparse
from sklearn.datasets import load_iris, make_regression, make_sparse_uncorrelated
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import (
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_preprocess_data(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 200
    n_features = 2
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    expected_X_mean = np.mean(X, axis=0)
    expected_y_mean = np.mean(y, axis=0)
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(X, y, fit_intercept=False)
    assert_array_almost_equal(X_mean, np.zeros(n_features))
    assert_array_almost_equal(y_mean, 0)
    assert_array_almost_equal(X_scale, np.ones(n_features))
    assert_array_almost_equal(Xt, X)
    assert_array_almost_equal(yt, y)
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(X, y, fit_intercept=True)
    assert_array_almost_equal(X_mean, expected_X_mean)
    assert_array_almost_equal(y_mean, expected_y_mean)
    assert_array_almost_equal(X_scale, np.ones(n_features))
    assert_array_almost_equal(Xt, X - expected_X_mean)
    assert_array_almost_equal(yt, y - expected_y_mean)