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
def test_polynomial_feature_names():
    X = np.arange(30).reshape(10, 3)
    poly = PolynomialFeatures(degree=2, include_bias=True).fit(X)
    feature_names = poly.get_feature_names_out()
    assert_array_equal(['1', 'x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2'], feature_names)
    assert len(feature_names) == poly.transform(X).shape[1]
    poly = PolynomialFeatures(degree=3, include_bias=False).fit(X)
    feature_names = poly.get_feature_names_out(['a', 'b', 'c'])
    assert_array_equal(['a', 'b', 'c', 'a^2', 'a b', 'a c', 'b^2', 'b c', 'c^2', 'a^3', 'a^2 b', 'a^2 c', 'a b^2', 'a b c', 'a c^2', 'b^3', 'b^2 c', 'b c^2', 'c^3'], feature_names)
    assert len(feature_names) == poly.transform(X).shape[1]
    poly = PolynomialFeatures(degree=(2, 3), include_bias=False).fit(X)
    feature_names = poly.get_feature_names_out(['a', 'b', 'c'])
    assert_array_equal(['a^2', 'a b', 'a c', 'b^2', 'b c', 'c^2', 'a^3', 'a^2 b', 'a^2 c', 'a b^2', 'a b c', 'a c^2', 'b^3', 'b^2 c', 'b c^2', 'c^3'], feature_names)
    assert len(feature_names) == poly.transform(X).shape[1]
    poly = PolynomialFeatures(degree=(3, 3), include_bias=True, interaction_only=True).fit(X)
    feature_names = poly.get_feature_names_out(['a', 'b', 'c'])
    assert_array_equal(['1', 'a b c'], feature_names)
    assert len(feature_names) == poly.transform(X).shape[1]
    poly = PolynomialFeatures(degree=1, include_bias=True).fit(X)
    feature_names = poly.get_feature_names_out(['\x01F40D', '☮', 'א'])
    assert_array_equal(['1', '\x01F40D', '☮', 'א'], feature_names)