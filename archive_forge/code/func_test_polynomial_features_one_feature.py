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
@pytest.mark.parametrize('degree, include_bias, interaction_only, indices', [(3, True, False, slice(None, None)), (3, False, False, slice(1, None)), (3, True, True, [0, 1]), (3, False, True, [1]), ((2, 3), True, False, [0, 2, 3]), ((2, 3), False, False, [2, 3]), ((2, 3), True, True, [0]), ((2, 3), False, True, [])])
@pytest.mark.parametrize('X_container', [None] + CSR_CONTAINERS + CSC_CONTAINERS)
def test_polynomial_features_one_feature(single_feature_degree3, degree, include_bias, interaction_only, indices, X_container):
    """Test PolynomialFeatures on single feature up to degree 3."""
    X, P = single_feature_degree3
    if X_container is not None:
        X = X_container(X)
    tf = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=interaction_only).fit(X)
    out = tf.transform(X)
    if X_container is not None:
        out = out.toarray()
    assert_allclose(out, P[:, indices])
    if tf.n_output_features_ > 0:
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)