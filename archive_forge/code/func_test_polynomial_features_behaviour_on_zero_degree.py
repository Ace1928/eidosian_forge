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
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + CSC_CONTAINERS)
def test_polynomial_features_behaviour_on_zero_degree(sparse_container):
    """Check that PolynomialFeatures raises error when degree=0 and include_bias=False,
    and output a single constant column when include_bias=True
    """
    X = np.ones((10, 2))
    poly = PolynomialFeatures(degree=0, include_bias=False)
    err_msg = 'Setting degree to zero and include_bias to False would result in an empty output array.'
    with pytest.raises(ValueError, match=err_msg):
        poly.fit_transform(X)
    poly = PolynomialFeatures(degree=(0, 0), include_bias=False)
    err_msg = 'Setting both min_degree and max_degree to zero and include_bias to False would result in an empty output array.'
    with pytest.raises(ValueError, match=err_msg):
        poly.fit_transform(X)
    for _X in [X, sparse_container(X)]:
        poly = PolynomialFeatures(degree=0, include_bias=True)
        output = poly.fit_transform(_X)
        if sparse.issparse(output):
            output = output.toarray()
        assert_array_equal(output, np.ones((X.shape[0], 1)))