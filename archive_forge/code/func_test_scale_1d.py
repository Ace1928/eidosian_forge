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
def test_scale_1d():
    X_list = [1.0, 3.0, 5.0, 0.0]
    X_arr = np.array(X_list)
    for X in [X_list, X_arr]:
        X_scaled = scale(X)
        assert_array_almost_equal(X_scaled.mean(), 0.0)
        assert_array_almost_equal(X_scaled.std(), 1.0)
        assert_array_equal(scale(X, with_mean=False, with_std=False), X)