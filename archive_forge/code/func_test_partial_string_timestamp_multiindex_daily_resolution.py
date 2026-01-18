import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_string_timestamp_multiindex_daily_resolution(df):
    result = df.loc[IndexSlice['2013-03':'2013-03', :], :]
    expected = df.iloc[118:180]
    tm.assert_frame_equal(result, expected)