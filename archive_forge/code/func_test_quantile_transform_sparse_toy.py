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
def test_quantile_transform_sparse_toy(csc_container):
    X = np.array([[0.0, 2.0, 0.0], [25.0, 4.0, 0.0], [50.0, 0.0, 2.6], [0.0, 0.0, 4.1], [0.0, 6.0, 0.0], [0.0, 8.0, 0.0], [75.0, 0.0, 2.3], [0.0, 10.0, 0.0], [0.0, 0.0, 9.5], [100.0, 0.0, 0.1]])
    X = csc_container(X)
    transformer = QuantileTransformer(n_quantiles=10)
    transformer.fit(X)
    X_trans = transformer.fit_transform(X)
    assert_array_almost_equal(np.min(X_trans.toarray(), axis=0), 0.0)
    assert_array_almost_equal(np.max(X_trans.toarray(), axis=0), 1.0)
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X.toarray(), X_trans_inv.toarray())
    transformer_dense = QuantileTransformer(n_quantiles=10).fit(X.toarray())
    X_trans = transformer_dense.transform(X)
    assert_array_almost_equal(np.min(X_trans.toarray(), axis=0), 0.0)
    assert_array_almost_equal(np.max(X_trans.toarray(), axis=0), 1.0)
    X_trans_inv = transformer_dense.inverse_transform(X_trans)
    assert_array_almost_equal(X.toarray(), X_trans_inv.toarray())