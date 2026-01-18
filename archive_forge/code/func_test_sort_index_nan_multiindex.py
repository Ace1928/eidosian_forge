import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_nan_multiindex(self):
    tuples = [[12, 13], [np.nan, np.nan], [np.nan, 3], [1, 2]]
    mi = MultiIndex.from_tuples(tuples)
    df = DataFrame(np.arange(16).reshape(4, 4), index=mi, columns=list('ABCD'))
    s = Series(np.arange(4), index=mi)
    df2 = DataFrame({'date': pd.DatetimeIndex(['20121002', '20121007', '20130130', '20130202', '20130305', '20121002', '20121207', '20130130', '20130202', '20130305', '20130202', '20130305']), 'user_id': [1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5, 5], 'whole_cost': [1790, np.nan, 280, 259, np.nan, 623, 90, 312, np.nan, 301, 359, 801], 'cost': [12, 15, 10, 24, 39, 1, 0, np.nan, 45, 34, 1, 12]}).set_index(['date', 'user_id'])
    result = df.sort_index()
    expected = df.iloc[[3, 0, 2, 1], :]
    tm.assert_frame_equal(result, expected)
    result = df.sort_index(na_position='last')
    expected = df.iloc[[3, 0, 2, 1], :]
    tm.assert_frame_equal(result, expected)
    result = df.sort_index(na_position='first')
    expected = df.iloc[[1, 2, 3, 0], :]
    tm.assert_frame_equal(result, expected)
    result = df2.dropna().sort_index()
    expected = df2.sort_index().dropna()
    tm.assert_frame_equal(result, expected)
    result = s.sort_index()
    expected = s.iloc[[3, 0, 2, 1]]
    tm.assert_series_equal(result, expected)
    result = s.sort_index(na_position='last')
    expected = s.iloc[[3, 0, 2, 1]]
    tm.assert_series_equal(result, expected)
    result = s.sort_index(na_position='first')
    expected = s.iloc[[1, 2, 3, 0]]
    tm.assert_series_equal(result, expected)