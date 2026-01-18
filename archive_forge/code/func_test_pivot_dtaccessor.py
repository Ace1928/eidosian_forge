from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_dtaccessor(self):
    dates1 = pd.DatetimeIndex(['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00', '2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'])
    dates2 = pd.DatetimeIndex(['2013-01-01 15:00:00', '2013-01-01 15:00:00', '2013-01-01 15:00:00', '2013-02-01 15:00:00', '2013-02-01 15:00:00', '2013-02-01 15:00:00'])
    df = DataFrame({'label': ['a', 'a', 'a', 'b', 'b', 'b'], 'dt1': dates1, 'dt2': dates2, 'value1': np.arange(6, dtype='int64'), 'value2': [1, 2] * 3})
    result = pivot_table(df, index='label', columns=df['dt1'].dt.hour, values='value1')
    exp_idx = Index(['a', 'b'], name='label')
    expected = DataFrame({7: [0.0, 3.0], 8: [1.0, 4.0], 9: [2.0, 5.0]}, index=exp_idx, columns=Index([7, 8, 9], dtype=np.int32, name='dt1'))
    tm.assert_frame_equal(result, expected)
    result = pivot_table(df, index=df['dt2'].dt.month, columns=df['dt1'].dt.hour, values='value1')
    expected = DataFrame({7: [0.0, 3.0], 8: [1.0, 4.0], 9: [2.0, 5.0]}, index=Index([1, 2], dtype=np.int32, name='dt2'), columns=Index([7, 8, 9], dtype=np.int32, name='dt1'))
    tm.assert_frame_equal(result, expected)
    result = pivot_table(df, index=df['dt2'].dt.year.values, columns=[df['dt1'].dt.hour, df['dt2'].dt.month], values='value1')
    exp_col = MultiIndex.from_arrays([np.array([7, 7, 8, 8, 9, 9], dtype=np.int32), np.array([1, 2] * 3, dtype=np.int32)], names=['dt1', 'dt2'])
    expected = DataFrame(np.array([[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]]), index=Index([2013], dtype=np.int32), columns=exp_col)
    tm.assert_frame_equal(result, expected)
    result = pivot_table(df, index=np.array(['X', 'X', 'X', 'X', 'Y', 'Y']), columns=[df['dt1'].dt.hour, df['dt2'].dt.month], values='value1')
    expected = DataFrame(np.array([[0, 3, 1, np.nan, 2, np.nan], [np.nan, np.nan, np.nan, 4, np.nan, 5]]), index=['X', 'Y'], columns=exp_col)
    tm.assert_frame_equal(result, expected)