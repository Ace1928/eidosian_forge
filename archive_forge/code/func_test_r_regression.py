import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('center', [True, False])
def test_r_regression(center):
    X, y = make_regression(n_samples=2000, n_features=20, n_informative=5, shuffle=False, random_state=0)
    corr_coeffs = r_regression(X, y, center=center)
    assert (-1 < corr_coeffs).all()
    assert (corr_coeffs < 1).all()
    sparse_X = _convert_container(X, 'sparse')
    sparse_corr_coeffs = r_regression(sparse_X, y, center=center)
    assert_allclose(sparse_corr_coeffs, corr_coeffs)
    Z = np.hstack((X, y[:, np.newaxis]))
    correlation_matrix = np.corrcoef(Z, rowvar=False)
    np_corr_coeffs = correlation_matrix[:-1, -1]
    assert_array_almost_equal(np_corr_coeffs, corr_coeffs, decimal=3)