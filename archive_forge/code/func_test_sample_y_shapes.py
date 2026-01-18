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
def test_sample_y_shapes(normalize_y, n_targets):
    """Check the shapes of y_samples in single-output (n_targets=0) and
    multi-output settings, including the edge case when n_targets=1, where the
    sklearn convention is to squeeze the predictions.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/22175
    """
    rng = np.random.RandomState(1234)
    n_features, n_samples_train = (6, 9)
    n_samples_X_test = 7
    n_samples_y_test = 5
    y_train_shape = (n_samples_train,)
    if n_targets is not None:
        y_train_shape = y_train_shape + (n_targets,)
    if n_targets is not None and n_targets > 1:
        y_test_shape = (n_samples_X_test, n_targets, n_samples_y_test)
    else:
        y_test_shape = (n_samples_X_test, n_samples_y_test)
    X_train = rng.randn(n_samples_train, n_features)
    X_test = rng.randn(n_samples_X_test, n_features)
    y_train = rng.randn(*y_train_shape)
    model = GaussianProcessRegressor(normalize_y=normalize_y)
    model.fit(X_train, y_train)
    y_samples = model.sample_y(X_test, n_samples=n_samples_y_test)
    assert y_samples.shape == y_test_shape