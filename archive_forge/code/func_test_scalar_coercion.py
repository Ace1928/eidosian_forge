from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('scalar', scalar_instances())
def test_scalar_coercion(self, scalar):
    if isinstance(scalar, np.inexact):
        scalar = type(scalar)((scalar * 2) ** 0.5)
    if type(scalar) is rational:
        pytest.xfail('Rational to object cast is undefined currently.')
    arr = np.array(scalar, dtype=object).astype(scalar.dtype)
    arr1 = np.array(scalar).reshape(1)
    arr2 = np.array([scalar])
    arr3 = np.empty(1, dtype=scalar.dtype)
    arr3[0] = scalar
    arr4 = np.empty(1, dtype=scalar.dtype)
    arr4[:] = [scalar]
    assert_array_equal(arr, arr1)
    assert_array_equal(arr, arr2)
    assert_array_equal(arr, arr3)
    assert_array_equal(arr, arr4)