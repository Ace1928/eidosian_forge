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
@pytest.mark.parametrize('lil_container', LIL_CONTAINERS)
def test_sparse_preprocess_data_offsets(global_random_seed, lil_container):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 200
    n_features = 2
    X = sparse.rand(n_samples, n_features, density=0.5, random_state=rng)
    X = lil_container(X)
    y = rng.rand(n_samples)
    XA = X.toarray()
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(X, y, fit_intercept=False)
    assert_array_almost_equal(X_mean, np.zeros(n_features))
    assert_array_almost_equal(y_mean, 0)
    assert_array_almost_equal(X_scale, np.ones(n_features))
    assert_array_almost_equal(Xt.toarray(), XA)
    assert_array_almost_equal(yt, y)
    Xt, yt, X_mean, y_mean, X_scale = _preprocess_data(X, y, fit_intercept=True)
    assert_array_almost_equal(X_mean, np.mean(XA, axis=0))
    assert_array_almost_equal(y_mean, np.mean(y, axis=0))
    assert_array_almost_equal(X_scale, np.ones(n_features))
    assert_array_almost_equal(Xt.toarray(), XA)
    assert_array_almost_equal(yt, y - np.mean(y, axis=0))