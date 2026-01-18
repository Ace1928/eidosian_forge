import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils._encode import _check_unknown, _encode, _get_counts, _unique
@pytest.mark.parametrize('missing_value', [np.nan, None, float('nan')])
@pytest.mark.parametrize('pickle_uniques', [True, False])
def test_unique_util_missing_values_objects(missing_value, pickle_uniques):
    values = np.array(['a', 'c', 'c', missing_value, 'b'], dtype=object)
    expected_uniques = np.array(['a', 'b', 'c', missing_value], dtype=object)
    uniques = _unique(values)
    if missing_value is None:
        assert_array_equal(uniques, expected_uniques)
    else:
        assert_array_equal(uniques[:-1], expected_uniques[:-1])
        assert np.isnan(uniques[-1])
    if pickle_uniques:
        uniques = pickle.loads(pickle.dumps(uniques))
    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, np.array([0, 2, 2, 3, 1]))