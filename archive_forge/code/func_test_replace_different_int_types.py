import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_different_int_types(self, any_int_numpy_dtype):
    labs = pd.Series([1, 1, 1, 0, 0, 2, 2, 2], dtype=any_int_numpy_dtype)
    maps = pd.Series([0, 2, 1], dtype=any_int_numpy_dtype)
    map_dict = dict(zip(maps.values, maps.index))
    result = labs.replace(map_dict)
    expected = labs.replace({0: 0, 2: 1, 1: 2})
    tm.assert_series_equal(result, expected)