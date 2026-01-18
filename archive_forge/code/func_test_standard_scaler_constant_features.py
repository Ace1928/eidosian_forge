import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('scaler', [StandardScaler(with_mean=False), RobustScaler(with_centering=False)])
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize('add_sample_weight', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('constant', [0, 1.0, 100.0])
def test_standard_scaler_constant_features(scaler, add_sample_weight, sparse_container, dtype, constant):
    if isinstance(scaler, RobustScaler) and add_sample_weight:
        pytest.skip(f'{scaler.__class__.__name__} does not yet support sample_weight')
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 1
    if add_sample_weight:
        fit_params = dict(sample_weight=rng.uniform(size=n_samples) * 2)
    else:
        fit_params = {}
    X_array = np.full(shape=(n_samples, n_features), fill_value=constant, dtype=dtype)
    X = X_array if sparse_container is None else sparse_container(X_array)
    X_scaled = scaler.fit(X, **fit_params).transform(X)
    if isinstance(scaler, StandardScaler):
        assert_allclose(scaler.var_, np.zeros(X.shape[1]), atol=1e-07)
    assert_allclose(scaler.scale_, np.ones(X.shape[1]))
    assert X_scaled is not X
    assert_allclose_dense_sparse(X_scaled, X)
    if isinstance(scaler, StandardScaler) and (not add_sample_weight):
        X_scaled_2 = scale(X, with_mean=scaler.with_mean)
        assert X_scaled_2 is not X
        assert_allclose_dense_sparse(X_scaled_2, X)