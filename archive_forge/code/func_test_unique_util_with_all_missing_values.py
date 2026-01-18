import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils._encode import _check_unknown, _encode, _get_counts, _unique
def test_unique_util_with_all_missing_values():
    values = np.array([np.nan, 'a', 'c', 'c', None, float('nan'), None], dtype=object)
    uniques = _unique(values)
    assert_array_equal(uniques[:-1], ['a', 'c', None])
    assert np.isnan(uniques[-1])
    expected_inverse = [3, 0, 1, 1, 2, 3, 2]
    _, inverse = _unique(values, return_inverse=True)
    assert_array_equal(inverse, expected_inverse)