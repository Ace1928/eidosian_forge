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
def test_robust_scaler_transform_one_row_csr(csr_container):
    rng = np.random.RandomState(0)
    X = rng.randn(4, 5)
    single_row = np.array([[0.1, 1.0, 2.0, 0.0, -1.0]])
    scaler = RobustScaler(with_centering=False)
    scaler = scaler.fit(X)
    row_trans = scaler.transform(csr_container(single_row))
    row_expected = single_row / scaler.scale_
    assert_array_almost_equal(row_trans.toarray(), row_expected)
    row_scaled_back = scaler.inverse_transform(row_trans)
    assert_array_almost_equal(single_row, row_scaled_back.toarray())