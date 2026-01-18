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
def test_prior(kernel):
    gpr = GaussianProcessRegressor(kernel=kernel)
    y_mean, y_cov = gpr.predict(X, return_cov=True)
    assert_almost_equal(y_mean, 0, 5)
    if len(gpr.kernel.theta) > 1:
        assert_almost_equal(np.diag(y_cov), np.exp(kernel.theta[0]), 5)
    else:
        assert_almost_equal(np.diag(y_cov), 1, 5)