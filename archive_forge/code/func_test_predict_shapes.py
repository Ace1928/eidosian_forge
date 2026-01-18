import re
import sys
import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import (
@pytest.mark.parametrize('normalize_y', [True, False])
@pytest.mark.parametrize('n_targets', [None, 1, 10])
def test_predict_shapes(normalize_y, n_targets):
    """Check the shapes of y_mean, y_std, and y_cov in single-output
    (n_targets=None) and multi-output settings, including the edge case when
    n_targets=1, where the sklearn convention is to squeeze the predictions.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/17394
    https://github.com/scikit-learn/scikit-learn/issues/18065
    https://github.com/scikit-learn/scikit-learn/issues/22174
    """
    rng = np.random.RandomState(1234)
    n_features, n_samples_train, n_samples_test = (6, 9, 7)
    y_train_shape = (n_samples_train,)
    if n_targets is not None:
        y_train_shape = y_train_shape + (n_targets,)
    y_test_shape = (n_samples_test,)
    if n_targets is not None and n_targets > 1:
        y_test_shape = y_test_shape + (n_targets,)
    X_train = rng.randn(n_samples_train, n_features)
    X_test = rng.randn(n_samples_test, n_features)
    y_train = rng.randn(*y_train_shape)
    model = GaussianProcessRegressor(normalize_y=normalize_y)
    model.fit(X_train, y_train)
    y_pred, y_std = model.predict(X_test, return_std=True)
    _, y_cov = model.predict(X_test, return_cov=True)
    assert y_pred.shape == y_test_shape
    assert y_std.shape == y_test_shape
    assert y_cov.shape == (n_samples_test,) + y_test_shape