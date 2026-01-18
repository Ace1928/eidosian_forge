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
@pytest.mark.parametrize('norm', ['l1', 'l2', 'max'])
@pytest.mark.parametrize('sparse_container', COO_CONTAINERS + CSC_CONTAINERS + LIL_CONTAINERS)
def test_normalizer_l1_l2_max_non_csr(norm, sparse_container):
    rng = np.random.RandomState(0)
    X_dense = rng.randn(4, 5)
    X_dense[3, :] = 0.0
    X = sparse_container(X_dense)
    X_norm = Normalizer(norm=norm, copy=False).transform(X)
    assert X_norm is not X
    assert sparse.issparse(X_norm) and X_norm.format == 'csr'
    X_norm = toarray(X_norm)
    check_normalizer(norm, X_norm)