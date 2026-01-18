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
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + CSC_CONTAINERS)
def test_scale_sparse_with_mean_raise_exception(sparse_container):
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X_sparse = sparse_container(X)
    with pytest.raises(ValueError):
        scale(X_sparse, with_mean=True)
    with pytest.raises(ValueError):
        StandardScaler(with_mean=True).fit(X_sparse)
    scaler = StandardScaler(with_mean=True).fit(X)
    with pytest.raises(ValueError):
        scaler.transform(X_sparse)
    X_transformed_sparse = sparse_container(scaler.transform(X))
    with pytest.raises(ValueError):
        scaler.inverse_transform(X_transformed_sparse)