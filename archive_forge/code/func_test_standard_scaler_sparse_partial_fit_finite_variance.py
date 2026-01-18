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
@pytest.mark.parametrize('X_2', [sparse.random(10, 1, density=0.8, random_state=0)] + [csr_container(np.full((10, 1), fill_value=np.nan)) for csr_container in CSR_CONTAINERS])
def test_standard_scaler_sparse_partial_fit_finite_variance(X_2):
    X_1 = sparse.random(5, 1, density=0.8)
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_1).partial_fit(X_2)
    assert np.isfinite(scaler.var_[0])