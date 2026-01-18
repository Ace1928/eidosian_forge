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
@pytest.mark.parametrize('method', ['box-cox', 'yeo-johnson'])
def test_power_transformer_nans(method):
    X = np.abs(X_1col)
    pt = PowerTransformer(method=method)
    pt.fit(X)
    lmbda_no_nans = pt.lambdas_[0]
    X = np.concatenate([X, np.full_like(X, np.nan)])
    X = shuffle(X, random_state=0)
    pt.fit(X)
    lmbda_nans = pt.lambdas_[0]
    assert_almost_equal(lmbda_no_nans, lmbda_nans, decimal=5)
    X_trans = pt.transform(X)
    assert_array_equal(np.isnan(X_trans), np.isnan(X))