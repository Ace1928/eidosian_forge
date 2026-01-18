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
def test_robust_scaler_col_zero_sparse(csr_container):
    X = np.random.randn(10, 5)
    X[:, 0] = 0
    X = csr_container(X)
    scaler = RobustScaler(with_centering=False)
    scaler.fit(X)
    assert scaler.scale_[0] == pytest.approx(1)
    X_trans = scaler.transform(X)
    assert_allclose(X[:, [0]].toarray(), X_trans[:, [0]].toarray())