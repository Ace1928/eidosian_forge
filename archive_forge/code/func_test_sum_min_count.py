import numpy as np
import pytest
from pandas import (
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('arr', [np.array([0, 1, np.nan, 1]), np.array([0, 1, 1])])
@pytest.mark.parametrize('fill_value', [0, 1, np.nan])
@pytest.mark.parametrize('min_count, expected', [(3, 2), (4, np.nan)])
def test_sum_min_count(self, arr, fill_value, min_count, expected):
    sparray = SparseArray(arr, fill_value=fill_value)
    result = sparray.sum(min_count=min_count)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected