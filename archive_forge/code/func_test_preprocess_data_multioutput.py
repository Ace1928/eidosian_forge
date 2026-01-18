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
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS)
def test_preprocess_data_multioutput(global_random_seed, sparse_container):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 200
    n_features = 3
    n_outputs = 2
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_outputs)
    expected_y_mean = np.mean(y, axis=0)
    if sparse_container is not None:
        X = sparse_container(X)
    _, yt, _, y_mean, _ = _preprocess_data(X, y, fit_intercept=False)
    assert_array_almost_equal(y_mean, np.zeros(n_outputs))
    assert_array_almost_equal(yt, y)
    _, yt, _, y_mean, _ = _preprocess_data(X, y, fit_intercept=True)
    assert_array_almost_equal(y_mean, expected_y_mean)
    assert_array_almost_equal(yt, y - y_mean)