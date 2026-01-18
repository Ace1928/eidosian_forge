import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils._encode import _check_unknown, _encode, _get_counts, _unique
def test_unique_util_missing_values_numeric():
    values = np.array([3, 1, np.nan, 5, 3, np.nan], dtype=float)
    expected_uniques = np.array([1, 3, 5, np.nan], dtype=float)
    expected_inverse = np.array([1, 0, 3, 2, 1, 3])
    uniques = _unique(values)
    assert_array_equal(uniques, expected_uniques)
    uniques, inverse = _unique(values, return_inverse=True)
    assert_array_equal(uniques, expected_uniques)
    assert_array_equal(inverse, expected_inverse)
    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, expected_inverse)