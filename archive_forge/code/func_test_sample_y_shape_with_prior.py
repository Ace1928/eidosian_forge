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
@pytest.mark.parametrize('n_targets', [None, 1, 2, 3])
@pytest.mark.parametrize('n_samples', [1, 5])
def test_sample_y_shape_with_prior(n_targets, n_samples):
    """Check the output shape of `sample_y` is consistent before and after `fit`."""
    rng = np.random.RandomState(1024)
    X = rng.randn(10, 3)
    y = rng.randn(10, n_targets if n_targets is not None else 1)
    model = GaussianProcessRegressor(n_targets=n_targets)
    shape_before_fit = model.sample_y(X, n_samples=n_samples).shape
    model.fit(X, y)
    shape_after_fit = model.sample_y(X, n_samples=n_samples).shape
    assert shape_before_fit == shape_after_fit