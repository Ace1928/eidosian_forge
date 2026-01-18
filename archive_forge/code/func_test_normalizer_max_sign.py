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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_normalizer_max_sign(csr_container):
    rng = np.random.RandomState(0)
    X_dense = rng.randn(4, 5)
    X_dense[3, :] = 0.0
    X_dense[2, abs(X_dense[2, :]).argmax()] *= -1
    X_all_neg = -np.abs(X_dense)
    X_all_neg_sparse = csr_container(X_all_neg)
    for X in (X_dense, X_all_neg, X_all_neg_sparse):
        normalizer = Normalizer(norm='max')
        X_norm = normalizer.transform(X)
        assert X_norm is not X
        X_norm = toarray(X_norm)
        assert_array_equal(np.sign(X_norm), np.sign(toarray(X)))