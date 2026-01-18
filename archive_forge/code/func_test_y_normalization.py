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
@pytest.mark.parametrize('kernel', kernels)
def test_y_normalization(kernel):
    """
    Test normalization of the target values in GP

    Fitting non-normalizing GP on normalized y and fitting normalizing GP
    on unnormalized y should yield identical results. Note that, here,
    'normalized y' refers to y that has been made zero mean and unit
    variance.

    """
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X, y_norm)
    gpr_norm = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr_norm.fit(X, y)
    y_pred, y_pred_std = gpr.predict(X2, return_std=True)
    y_pred = y_pred * y_std + y_mean
    y_pred_std = y_pred_std * y_std
    y_pred_norm, y_pred_std_norm = gpr_norm.predict(X2, return_std=True)
    assert_almost_equal(y_pred, y_pred_norm)
    assert_almost_equal(y_pred_std, y_pred_std_norm)
    _, y_cov = gpr.predict(X2, return_cov=True)
    y_cov = y_cov * y_std ** 2
    _, y_cov_norm = gpr_norm.predict(X2, return_cov=True)
    assert_almost_equal(y_cov, y_cov_norm)