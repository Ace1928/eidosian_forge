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
@pytest.mark.parametrize(['Xw', 'X', 'sample_weight'], [([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [1, 2, 3], [4, 5, 6]], [2.0, 1.0]), ([[1, 0, 1], [0, 0, 1]], [[1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], np.array([1, 3])), ([[1, np.nan, 1], [np.nan, np.nan, 1]], [[1, np.nan, 1], [np.nan, np.nan, 1], [np.nan, np.nan, 1], [np.nan, np.nan, 1]], np.array([1, 3]))])
@pytest.mark.parametrize('array_constructor', ['array', 'sparse_csr', 'sparse_csc'])
def test_standard_scaler_sample_weight(Xw, X, sample_weight, array_constructor):
    with_mean = not array_constructor.startswith('sparse')
    X = _convert_container(X, array_constructor)
    Xw = _convert_container(Xw, array_constructor)
    yw = np.ones(Xw.shape[0])
    scaler_w = StandardScaler(with_mean=with_mean)
    scaler_w.fit(Xw, yw, sample_weight=sample_weight)
    y = np.ones(X.shape[0])
    scaler = StandardScaler(with_mean=with_mean)
    scaler.fit(X, y)
    X_test = [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]
    assert_almost_equal(scaler.mean_, scaler_w.mean_)
    assert_almost_equal(scaler.var_, scaler_w.var_)
    assert_almost_equal(scaler.transform(X_test), scaler_w.transform(X_test))