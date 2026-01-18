import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_set_empty_frame_row(self):
    expected = DataFrame(columns=['A', 'B', 'New'], index=Index([], dtype='int64'))
    expected['A'] = expected['A'].astype('int64')
    expected['B'] = expected['B'].astype('float64')
    expected['New'] = expected['New'].astype('float64')
    df = DataFrame({'A': [1, 2, 3], 'B': [1.2, 4.2, 5.2]})
    y = df[df.A > 5]
    y['New'] = np.nan
    tm.assert_frame_equal(y, expected)
    expected = DataFrame(columns=['a', 'b', 'c c', 'd'])
    expected['d'] = expected['d'].astype('int64')
    df = DataFrame(columns=['a', 'b', 'c c'])
    df['d'] = 3
    tm.assert_frame_equal(df, expected)
    tm.assert_series_equal(df['c c'], Series(name='c c', dtype=object))
    df = DataFrame({'A': [1, 2, 3], 'B': [1.2, 4.2, 5.2]})
    y = df[df.A > 5]
    result = y.reindex(columns=['A', 'B', 'C'])
    expected = DataFrame(columns=['A', 'B', 'C'])
    expected['A'] = expected['A'].astype('int64')
    expected['B'] = expected['B'].astype('float64')
    expected['C'] = expected['C'].astype('float64')
    tm.assert_frame_equal(result, expected)