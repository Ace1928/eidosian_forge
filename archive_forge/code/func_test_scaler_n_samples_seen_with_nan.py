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
@pytest.mark.parametrize('with_mean', [True, False])
@pytest.mark.parametrize('with_std', [True, False])
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_scaler_n_samples_seen_with_nan(with_mean, with_std, sparse_container):
    X = np.array([[0, 1, 3], [np.nan, 6, 10], [5, 4, np.nan], [8, 0, np.nan]], dtype=np.float64)
    if sparse_container is not None:
        X = sparse_container(X)
    if sparse.issparse(X) and with_mean:
        pytest.skip("'with_mean=True' cannot be used with sparse matrix.")
    transformer = StandardScaler(with_mean=with_mean, with_std=with_std)
    transformer.fit(X)
    assert_array_equal(transformer.n_samples_seen_, np.array([3, 4, 2]))