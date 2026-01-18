from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def test_subarray_from_array_construction():
    arr = np.array([1, 2])
    res = arr.astype('(2)i,')
    assert_array_equal(res, [[1, 1], [2, 2]])
    res = np.array(arr, dtype='(2)i,')
    assert_array_equal(res, [[1, 1], [2, 2]])
    res = np.array([[(1,), (2,)], arr], dtype='(2)i,')
    assert_array_equal(res, [[[1, 1], [2, 2]], [[1, 1], [2, 2]]])
    arr = np.arange(5 * 2).reshape(5, 2)
    expected = np.broadcast_to(arr[:, :, np.newaxis, np.newaxis], (5, 2, 2, 2))
    res = arr.astype('(2,2)f')
    assert_array_equal(res, expected)
    res = np.array(arr, dtype='(2,2)f')
    assert_array_equal(res, expected)