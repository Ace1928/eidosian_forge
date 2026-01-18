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
@pytest.mark.parametrize('X_shape', [(11, 8), (11, 20)])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('y_shape, noise', [((11,), 1.0), ((11, 1), 30.0), ((11, 3), 150.0)])
def test_ridge_gcv_vs_ridge_loo_cv(gcv_mode, X_container, X_shape, y_shape, fit_intercept, noise):
    n_samples, n_features = X_shape
    n_targets = y_shape[-1] if len(y_shape) == 2 else 1
    X, y = _make_sparse_offset_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, random_state=0, shuffle=False, noise=noise, n_informative=5)
    y = y.reshape(y_shape)
    alphas = [0.001, 0.1, 1.0, 10.0, 1000.0]
    loo_ridge = RidgeCV(cv=n_samples, fit_intercept=fit_intercept, alphas=alphas, scoring='neg_mean_squared_error')
    gcv_ridge = RidgeCV(gcv_mode=gcv_mode, fit_intercept=fit_intercept, alphas=alphas)
    loo_ridge.fit(X, y)
    X_gcv = X_container(X)
    gcv_ridge.fit(X_gcv, y)
    assert gcv_ridge.alpha_ == pytest.approx(loo_ridge.alpha_)
    assert_allclose(gcv_ridge.coef_, loo_ridge.coef_, rtol=0.001)
    assert_allclose(gcv_ridge.intercept_, loo_ridge.intercept_, rtol=0.001)