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
def test_standard_scaler_raise_error_for_1d_input():
    """Check that `inverse_transform` from `StandardScaler` raises an error
    with 1D array.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19518
    """
    scaler = StandardScaler().fit(X_2d)
    err_msg = 'Expected 2D array, got 1D array instead'
    with pytest.raises(ValueError, match=err_msg):
        scaler.inverse_transform(X_2d[:, 0])