import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
def test_warning_bounds():
    kernel = RBF(length_scale_bounds=[1e-05, 0.001])
    gpc = GaussianProcessClassifier(kernel=kernel)
    warning_message = 'The optimal value found for dimension 0 of parameter length_scale is close to the specified upper bound 0.001. Increasing the bound and calling fit again may find a better value.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        gpc.fit(X, y)
    kernel_sum = WhiteKernel(noise_level_bounds=[1e-05, 0.001]) + RBF(length_scale_bounds=[1000.0, 100000.0])
    gpc_sum = GaussianProcessClassifier(kernel=kernel_sum)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        gpc_sum.fit(X, y)
        assert len(record) == 2
        assert issubclass(record[0].category, ConvergenceWarning)
        assert record[0].message.args[0] == 'The optimal value found for dimension 0 of parameter k1__noise_level is close to the specified upper bound 0.001. Increasing the bound and calling fit again may find a better value.'
        assert issubclass(record[1].category, ConvergenceWarning)
        assert record[1].message.args[0] == 'The optimal value found for dimension 0 of parameter k2__length_scale is close to the specified lower bound 1000.0. Decreasing the bound and calling fit again may find a better value.'
    X_tile = np.tile(X, 2)
    kernel_dims = RBF(length_scale=[1.0, 2.0], length_scale_bounds=[10.0, 100.0])
    gpc_dims = GaussianProcessClassifier(kernel=kernel_dims)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        gpc_dims.fit(X_tile, y)
        assert len(record) == 2
        assert issubclass(record[0].category, ConvergenceWarning)
        assert record[0].message.args[0] == 'The optimal value found for dimension 0 of parameter length_scale is close to the specified upper bound 100.0. Increasing the bound and calling fit again may find a better value.'
        assert issubclass(record[1].category, ConvergenceWarning)
        assert record[1].message.args[0] == 'The optimal value found for dimension 1 of parameter length_scale is close to the specified upper bound 100.0. Increasing the bound and calling fit again may find a better value.'