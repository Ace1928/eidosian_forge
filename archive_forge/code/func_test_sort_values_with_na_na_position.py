import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList
@pytest.mark.parametrize('na_position', ['first', 'last'])
@pytest.mark.parametrize('dtype', ['float64', 'Int64', 'Float64'])
def test_sort_values_with_na_na_position(dtype, na_position):
    arrays = [Series([1, 1, 2], dtype=dtype), Series([1, None, 3], dtype=dtype)]
    index = MultiIndex.from_arrays(arrays)
    result = index.sort_values(na_position=na_position)
    if na_position == 'first':
        arrays = [Series([1, 1, 2], dtype=dtype), Series([None, 1, 3], dtype=dtype)]
    else:
        arrays = [Series([1, 1, 2], dtype=dtype), Series([1, None, 3], dtype=dtype)]
    expected = MultiIndex.from_arrays(arrays)
    tm.assert_index_equal(result, expected)