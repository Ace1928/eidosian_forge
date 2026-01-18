import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_xs_partial(self, multiindex_dataframe_random_data, multiindex_year_month_day_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    ymd = multiindex_year_month_day_dataframe_random_data
    result = frame.xs('foo')
    result2 = frame.loc['foo']
    expected = frame.T['foo'].T
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result, result2)
    result = ymd.xs((2000, 4))
    expected = ymd.loc[2000, 4]
    tm.assert_frame_equal(result, expected)
    index = MultiIndex(levels=[['foo', 'bar'], ['one', 'two'], [-1, 1]], codes=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]])
    df = DataFrame(np.random.default_rng(2).standard_normal((8, 4)), index=index, columns=list('abcd'))
    result = df.xs(('foo', 'one'))
    expected = df.loc['foo', 'one']
    tm.assert_frame_equal(result, expected)