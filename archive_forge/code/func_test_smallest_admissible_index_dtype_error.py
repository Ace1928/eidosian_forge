import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('params, err_type, err_msg', [({'maxval': np.iinfo(np.int64).max + 1}, ValueError, 'is to large to be represented as np.int64'), ({'arrays': np.array([1, 2], dtype=np.float64)}, ValueError, 'Array dtype float64 is not supported'), ({'arrays': [1, 2]}, TypeError, 'Arrays should be of type np.ndarray')])
def test_smallest_admissible_index_dtype_error(params, err_type, err_msg):
    """Check that we raise the proper error message."""
    with pytest.raises(err_type, match=err_msg):
        _smallest_admissible_index_dtype(**params)