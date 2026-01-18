import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_partial_set(self, multiindex_year_month_day_dataframe_random_data, using_copy_on_write, warn_copy_on_write):
    ymd = multiindex_year_month_day_dataframe_random_data
    df = ymd.copy()
    exp = ymd.copy()
    df.loc[2000, 4] = 0
    exp.iloc[65:85] = 0
    tm.assert_frame_equal(df, exp)
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['A'].loc[2000, 4] = 1
        df.loc[(2000, 4), 'A'] = 1
    else:
        with tm.raises_chained_assignment_error():
            df['A'].loc[2000, 4] = 1
    exp.iloc[65:85, 0] = 1
    tm.assert_frame_equal(df, exp)
    df.loc[2000] = 5
    exp.iloc[:100] = 5
    tm.assert_frame_equal(df, exp)
    with tm.raises_chained_assignment_error():
        df['A'].iloc[14] = 5
    if using_copy_on_write:
        assert df['A'].iloc[14] == exp['A'].iloc[14]
    else:
        assert df['A'].iloc[14] == 5