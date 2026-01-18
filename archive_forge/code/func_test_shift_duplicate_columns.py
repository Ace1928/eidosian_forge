import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_duplicate_columns(self):
    column_lists = [list(range(5)), [1] * 5, [1, 1, 2, 2, 1]]
    data = np.random.default_rng(2).standard_normal((20, 5))
    shifted = []
    for columns in column_lists:
        df = DataFrame(data.copy(), columns=columns)
        for s in range(5):
            df.iloc[:, s] = df.iloc[:, s].shift(s + 1)
        df.columns = range(5)
        shifted.append(df)
    nulls = shifted[0].isna().sum()
    tm.assert_series_equal(nulls, Series(range(1, 6), dtype='int64'))
    tm.assert_frame_equal(shifted[0], shifted[1])
    tm.assert_frame_equal(shifted[0], shifted[2])