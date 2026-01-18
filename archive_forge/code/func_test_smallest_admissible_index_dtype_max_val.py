import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('params, expected_dtype', [({}, np.int32), ({'maxval': np.iinfo(np.int32).max}, np.int32), ({'maxval': np.iinfo(np.int32).max + 1}, np.int64)])
def test_smallest_admissible_index_dtype_max_val(params, expected_dtype):
    """Check the behaviour of `smallest_admissible_index_dtype` depending only on the
    `max_val` parameter.
    """
    assert _smallest_admissible_index_dtype(**params) == expected_dtype