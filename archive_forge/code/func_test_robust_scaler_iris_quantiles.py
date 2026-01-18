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
def test_robust_scaler_iris_quantiles():
    X = iris.data
    scaler = RobustScaler(quantile_range=(10, 90))
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(np.median(X_trans, axis=0), 0)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    q = np.percentile(X_trans, q=(10, 90), axis=0)
    q_range = q[1] - q[0]
    assert_array_almost_equal(q_range, 1)