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
@pytest.mark.parametrize('sample_weight', [True, None])
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
def test_partial_fit_sparse_input(sample_weight, sparse_container):
    X = sparse_container(np.array([[1.0], [0.0], [0.0], [5.0]]))
    if sample_weight:
        sample_weight = rng.rand(X.shape[0])
    null_transform = StandardScaler(with_mean=False, with_std=False, copy=True)
    X_null = null_transform.partial_fit(X, sample_weight=sample_weight).transform(X)
    assert_array_equal(X_null.toarray(), X.toarray())
    X_orig = null_transform.inverse_transform(X_null)
    assert_array_equal(X_orig.toarray(), X_null.toarray())
    assert_array_equal(X_orig.toarray(), X.toarray())