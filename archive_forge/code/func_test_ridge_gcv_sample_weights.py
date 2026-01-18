import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('gcv_mode', ['svd', 'eigen'])
@pytest.mark.parametrize('X_container', [np.asarray] + CSR_CONTAINERS)
@pytest.mark.parametrize('n_features', [8, 20])
@pytest.mark.parametrize('y_shape, fit_intercept, noise', [((11,), True, 1.0), ((11, 1), True, 20.0), ((11, 3), True, 150.0), ((11, 3), False, 30.0)])
def test_ridge_gcv_sample_weights(gcv_mode, X_container, fit_intercept, n_features, y_shape, noise):
    alphas = [0.001, 0.1, 1.0, 10.0, 1000.0]
    rng = np.random.RandomState(0)
    n_targets = y_shape[-1] if len(y_shape) == 2 else 1
    X, y = _make_sparse_offset_regression(n_samples=11, n_features=n_features, n_targets=n_targets, random_state=0, shuffle=False, noise=noise)
    y = y.reshape(y_shape)
    sample_weight = 3 * rng.randn(len(X))
    sample_weight = (sample_weight - sample_weight.min() + 1).astype(int)
    indices = np.repeat(np.arange(X.shape[0]), sample_weight)
    sample_weight = sample_weight.astype(float)
    X_tiled, y_tiled = (X[indices], y[indices])
    cv = GroupKFold(n_splits=X.shape[0])
    splits = cv.split(X_tiled, y_tiled, groups=indices)
    kfold = RidgeCV(alphas=alphas, cv=splits, scoring='neg_mean_squared_error', fit_intercept=fit_intercept)
    kfold.fit(X_tiled, y_tiled)
    ridge_reg = Ridge(alpha=kfold.alpha_, fit_intercept=fit_intercept)
    splits = cv.split(X_tiled, y_tiled, groups=indices)
    predictions = cross_val_predict(ridge_reg, X_tiled, y_tiled, cv=splits)
    kfold_errors = (y_tiled - predictions) ** 2
    kfold_errors = [np.sum(kfold_errors[indices == i], axis=0) for i in np.arange(X.shape[0])]
    kfold_errors = np.asarray(kfold_errors)
    X_gcv = X_container(X)
    gcv_ridge = RidgeCV(alphas=alphas, store_cv_values=True, gcv_mode=gcv_mode, fit_intercept=fit_intercept)
    gcv_ridge.fit(X_gcv, y, sample_weight=sample_weight)
    if len(y_shape) == 2:
        gcv_errors = gcv_ridge.cv_values_[:, :, alphas.index(kfold.alpha_)]
    else:
        gcv_errors = gcv_ridge.cv_values_[:, alphas.index(kfold.alpha_)]
    assert kfold.alpha_ == pytest.approx(gcv_ridge.alpha_)
    assert_allclose(gcv_errors, kfold_errors, rtol=0.001)
    assert_allclose(gcv_ridge.coef_, kfold.coef_, rtol=0.001)
    assert_allclose(gcv_ridge.intercept_, kfold.intercept_, rtol=0.001)