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
def test_quantile_transform_and_inverse():
    X_1 = iris.data
    X_2 = np.array([[0.0], [BOUNDS_THRESHOLD / 10], [1.5], [2], [3], [3], [4]])
    for X in [X_1, X_2]:
        transformer = QuantileTransformer(n_quantiles=1000, random_state=0)
        X_trans = transformer.fit_transform(X)
        X_trans_inv = transformer.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv, decimal=9)