import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_string_timestamp_multiindex(df):
    df_swap = df.swaplevel(0, 1).sort_index()
    SLC = IndexSlice
    result = df.loc[SLC['2016-01-01':'2016-02-01', :], :]
    expected = df
    tm.assert_frame_equal(result, expected)
    result = df_swap.loc[SLC[:, '2016-01-01':'2016-01-01'], :]
    expected = df_swap.iloc[[0, 1, 5, 6, 10, 11]]
    tm.assert_frame_equal(result, expected)
    result = df.loc['2016']
    expected = df
    tm.assert_frame_equal(result, expected)
    result = df.loc['2016-01-01']
    expected = df.iloc[0:6]
    tm.assert_frame_equal(result, expected)
    result = df.loc['2016-01-02 12']
    expected = df.iloc[9:12].droplevel(0)
    tm.assert_frame_equal(result, expected)
    result = df_swap.loc[SLC[:, '2016-01-02'], :]
    expected = df_swap.iloc[[2, 3, 7, 8, 12, 13]]
    tm.assert_frame_equal(result, expected)
    result = df.loc[('2016-01-01', 'a'), :]
    expected = df.iloc[[0, 3]]
    expected = df.iloc[[0, 3]].droplevel(1)
    tm.assert_frame_equal(result, expected)
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df_swap.loc['2016-01-01']