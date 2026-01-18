import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_keep_ea_dtype_with_na(any_numeric_ea_dtype):
    arr1 = Series([4, pd.NA], dtype=any_numeric_ea_dtype)
    arr2 = Series([1, pd.NA], dtype=any_numeric_ea_dtype)
    midx = MultiIndex.from_arrays([arr1, [2, 1]], names=['a', None])
    midx2 = MultiIndex.from_arrays([arr2, [1, 2]])
    result = midx.union(midx2)
    expected = MultiIndex.from_arrays([Series([1, 4, pd.NA, pd.NA], dtype=any_numeric_ea_dtype), [1, 2, 1, 2]])
    tm.assert_index_equal(result, expected)