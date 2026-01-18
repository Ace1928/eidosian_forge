import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_array_no_sort():
    df = DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
    key = np.array([1, 0, 1], dtype=np.int64)
    result = df.groupby(key, sort=False).quantile([0.25, 0.5, 0.75])
    expected = DataFrame({'A': [0.5, 1.0, 1.5, 1.0, 1.0, 1.0], 'B': [3.5, 4.0, 4.5, 4.0, 4.0, 4.0]}, index=pd.MultiIndex.from_product([[1, 0], [0.25, 0.5, 0.75]]))
    tm.assert_frame_equal(result, expected)
    result = df.groupby(key, sort=False).quantile([0.75, 0.25])
    expected = DataFrame({'A': [1.5, 0.5, 1.0, 1.0], 'B': [4.5, 3.5, 4.0, 4.0]}, index=pd.MultiIndex.from_product([[1, 0], [0.75, 0.25]]))
    tm.assert_frame_equal(result, expected)