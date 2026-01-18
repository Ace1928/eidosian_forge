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
def test_quantile_transform_axis1():
    X = np.array([[0, 25, 50, 75, 100], [2, 4, 6, 8, 10], [2.6, 4.1, 2.3, 9.5, 0.1]])
    X_trans_a0 = quantile_transform(X.T, axis=0, n_quantiles=5)
    X_trans_a1 = quantile_transform(X, axis=1, n_quantiles=5)
    assert_array_almost_equal(X_trans_a0, X_trans_a1.T)