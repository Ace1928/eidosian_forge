from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_unstack_multiple(self, multiindex_year_month_day_dataframe_random_data, future_stack):
    ymd = multiindex_year_month_day_dataframe_random_data
    unstacked = ymd.unstack(['year', 'month'])
    expected = ymd.unstack('year').unstack('month')
    tm.assert_frame_equal(unstacked, expected)
    assert unstacked.columns.names == expected.columns.names
    s = ymd['A']
    s_unstacked = s.unstack(['year', 'month'])
    tm.assert_frame_equal(s_unstacked, expected['A'])
    restacked = unstacked.stack(['year', 'month'], future_stack=future_stack)
    if future_stack:
        restacked = restacked.dropna(how='all')
    restacked = restacked.swaplevel(0, 1).swaplevel(1, 2)
    restacked = restacked.sort_index(level=0)
    tm.assert_frame_equal(restacked, ymd)
    assert restacked.index.names == ymd.index.names
    unstacked = ymd.unstack([1, 2])
    expected = ymd.unstack(1).unstack(1).dropna(axis=1, how='all')
    tm.assert_frame_equal(unstacked, expected)
    unstacked = ymd.unstack([2, 1])
    expected = ymd.unstack(2).unstack(1).dropna(axis=1, how='all')
    tm.assert_frame_equal(unstacked, expected.loc[:, unstacked.columns])