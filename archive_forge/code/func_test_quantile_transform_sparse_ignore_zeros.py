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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_quantile_transform_sparse_ignore_zeros(csc_container):
    X = np.array([[0, 1], [0, 0], [0, 2], [0, 2], [0, 1]])
    X_sparse = csc_container(X)
    transformer = QuantileTransformer(ignore_implicit_zeros=True, n_quantiles=5)
    warning_message = "'ignore_implicit_zeros' takes effect only with sparse matrix. This parameter has no effect."
    with pytest.warns(UserWarning, match=warning_message):
        transformer.fit(X)
    X_expected = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]])
    X_trans = transformer.fit_transform(X_sparse)
    assert_almost_equal(X_expected, X_trans.toarray())
    X_data = np.array([0, 0, 1, 0, 2, 2, 1, 0, 1, 2, 0])
    X_col = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    X_row = np.array([0, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    X_sparse = csc_container((X_data, (X_row, X_col)))
    X_trans = transformer.fit_transform(X_sparse)
    X_expected = np.array([[0.0, 0.5], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.5], [0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [0.0, 0.0]])
    assert_almost_equal(X_expected, X_trans.toarray())
    transformer = QuantileTransformer(ignore_implicit_zeros=True, n_quantiles=5)
    X_data = np.array([-1, -1, 1, 0, 0, 0, 1, -1, 1])
    X_col = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1])
    X_row = np.array([0, 4, 0, 1, 2, 3, 4, 5, 6])
    X_sparse = csc_container((X_data, (X_row, X_col)))
    X_trans = transformer.fit_transform(X_sparse)
    X_expected = np.array([[0, 1], [0, 0.375], [0, 0.375], [0, 0.375], [0, 1], [0, 0], [0, 1]])
    assert_almost_equal(X_expected, X_trans.toarray())
    assert_almost_equal(X_sparse.toarray(), transformer.inverse_transform(X_trans).toarray())
    transformer = QuantileTransformer(ignore_implicit_zeros=True, n_quantiles=5, subsample=8, random_state=0)
    X_trans = transformer.fit_transform(X_sparse)
    assert_almost_equal(X_expected, X_trans.toarray())
    assert_almost_equal(X_sparse.toarray(), transformer.inverse_transform(X_trans).toarray())