import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose
from scipy import linalg
from scipy.optimize import minimize, root
from sklearn._loss import HalfBinomialLoss, HalfPoissonLoss, HalfTweedieLoss
from sklearn._loss.link import IdentityLink, LogLink
from sklearn.base import clone
from sklearn.datasets import make_low_rank_matrix, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._glm import _GeneralizedLinearRegressor
from sklearn.linear_model._glm._newton_solver import NewtonCholeskySolver
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance
from sklearn.model_selection import train_test_split
@pytest.mark.parametrize('n_samples, n_features', [(100, 10), (10, 100)])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('sample_weight', [None, True])
def test_normal_ridge_comparison(n_samples, n_features, fit_intercept, sample_weight, request):
    """Compare with Ridge regression for Normal distributions."""
    test_size = 10
    X, y = make_regression(n_samples=n_samples + test_size, n_features=n_features, n_informative=n_features - 2, noise=0.5, random_state=42)
    if n_samples > n_features:
        ridge_params = {'solver': 'svd'}
    else:
        ridge_params = {'solver': 'saga', 'max_iter': 1000000, 'tol': 1e-07}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    alpha = 1.0
    if sample_weight is None:
        sw_train = None
        alpha_ridge = alpha * n_samples
    else:
        sw_train = np.random.RandomState(0).rand(len(y_train))
        alpha_ridge = alpha * sw_train.sum()
    ridge = Ridge(alpha=alpha_ridge, random_state=42, fit_intercept=fit_intercept, **ridge_params)
    ridge.fit(X_train, y_train, sample_weight=sw_train)
    glm = _GeneralizedLinearRegressor(alpha=alpha, fit_intercept=fit_intercept, max_iter=300, tol=1e-05)
    glm.fit(X_train, y_train, sample_weight=sw_train)
    assert glm.coef_.shape == (X.shape[1],)
    assert_allclose(glm.coef_, ridge.coef_, atol=5e-05)
    assert_allclose(glm.intercept_, ridge.intercept_, rtol=1e-05)
    assert_allclose(glm.predict(X_train), ridge.predict(X_train), rtol=0.0002)
    assert_allclose(glm.predict(X_test), ridge.predict(X_test), rtol=0.0002)