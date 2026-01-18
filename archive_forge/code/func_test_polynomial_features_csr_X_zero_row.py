import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse import random as sparse_random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._csr_polynomial_expansion import (
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize(['zero_row_index', 'deg', 'interaction_only'], [(0, 2, True), (1, 2, True), (2, 2, True), (0, 3, True), (1, 3, True), (2, 3, True), (0, 2, False), (1, 2, False), (2, 2, False), (0, 3, False), (1, 3, False), (2, 3, False)])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_polynomial_features_csr_X_zero_row(zero_row_index, deg, interaction_only, csr_container):
    X_csr = csr_container(sparse_random(3, 10, 1.0, random_state=0))
    X_csr[zero_row_index, :] = 0.0
    X = X_csr.toarray()
    est = PolynomialFeatures(deg, include_bias=False, interaction_only=interaction_only)
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)
    assert sparse.issparse(Xt_csr) and Xt_csr.format == 'csr'
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)