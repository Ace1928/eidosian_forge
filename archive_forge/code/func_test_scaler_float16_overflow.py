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
def test_scaler_float16_overflow():
    rng = np.random.RandomState(0)
    X = rng.uniform(5, 10, [200000, 1]).astype(np.float16)
    with np.errstate(over='raise'):
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
    X_scaled_f64 = StandardScaler().fit_transform(X.astype(np.float64))
    assert np.all(np.isfinite(X_scaled))
    assert_array_almost_equal(X_scaled, X_scaled_f64, decimal=2)