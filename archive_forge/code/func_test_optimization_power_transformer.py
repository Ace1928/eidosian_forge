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
@pytest.mark.parametrize('method, lmbda', [('box-cox', 0.1), ('box-cox', 0.5), ('yeo-johnson', 0.1), ('yeo-johnson', 0.5), ('yeo-johnson', 1.0)])
def test_optimization_power_transformer(method, lmbda):
    rng = np.random.RandomState(0)
    n_samples = 20000
    X = rng.normal(loc=0, scale=1, size=(n_samples, 1))
    pt = PowerTransformer(method=method, standardize=False)
    pt.lambdas_ = [lmbda]
    X_inv = pt.inverse_transform(X)
    pt = PowerTransformer(method=method, standardize=False)
    X_inv_trans = pt.fit_transform(X_inv)
    assert_almost_equal(0, np.linalg.norm(X - X_inv_trans) / n_samples, decimal=2)
    assert_almost_equal(0, X_inv_trans.mean(), decimal=1)
    assert_almost_equal(1, X_inv_trans.std(), decimal=1)