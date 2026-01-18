import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('params, expected_dtype', [({'arrays': np.array([1, 2], dtype=np.int64)}, np.int64), ({'arrays': (np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int64))}, np.int64), ({'arrays': (np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int32))}, np.int32), ({'arrays': np.array([1, 2], dtype=np.int8)}, np.int32), ({'arrays': np.array([1, 2], dtype=np.int32), 'maxval': np.iinfo(np.int32).max + 1}, np.int64)])
def test_smallest_admissible_index_dtype_without_checking_contents(params, expected_dtype):
    """Check the behaviour of `smallest_admissible_index_dtype` using the passed
    arrays but without checking the contents of the arrays.
    """
    assert _smallest_admissible_index_dtype(**params) == expected_dtype