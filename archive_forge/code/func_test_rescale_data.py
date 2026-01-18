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
@pytest.mark.parametrize('n_targets', [None, 2])
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
def test_rescale_data(n_targets, sparse_container, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 200
    n_features = 2
    sample_weight = 1.0 + rng.rand(n_samples)
    X = rng.rand(n_samples, n_features)
    if n_targets is None:
        y = rng.rand(n_samples)
    else:
        y = rng.rand(n_samples, n_targets)
    expected_sqrt_sw = np.sqrt(sample_weight)
    expected_rescaled_X = X * expected_sqrt_sw[:, np.newaxis]
    if n_targets is None:
        expected_rescaled_y = y * expected_sqrt_sw
    else:
        expected_rescaled_y = y * expected_sqrt_sw[:, np.newaxis]
    if sparse_container is not None:
        X = sparse_container(X)
        if n_targets is None:
            y = sparse_container(y.reshape(-1, 1))
        else:
            y = sparse_container(y)
    rescaled_X, rescaled_y, sqrt_sw = _rescale_data(X, y, sample_weight)
    assert_allclose(sqrt_sw, expected_sqrt_sw)
    if sparse_container is not None:
        rescaled_X = rescaled_X.toarray()
        rescaled_y = rescaled_y.toarray()
        if n_targets is None:
            rescaled_y = rescaled_y.ravel()
    assert_allclose(rescaled_X, expected_rescaled_X)
    assert_allclose(rescaled_y, expected_rescaled_y)