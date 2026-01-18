import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_nat(self):
    d1 = [Timestamp(x) for x in ['2016-01-01', '2015-01-01', np.nan, '2016-01-01']]
    d2 = [Timestamp(x) for x in ['2017-01-01', '2014-01-01', '2016-01-01', '2015-01-01']]
    df = DataFrame({'a': d1, 'b': d2}, index=[0, 1, 2, 3])
    d3 = [Timestamp(x) for x in ['2015-01-01', '2016-01-01', '2016-01-01', np.nan]]
    d4 = [Timestamp(x) for x in ['2014-01-01', '2015-01-01', '2017-01-01', '2016-01-01']]
    expected = DataFrame({'a': d3, 'b': d4}, index=[1, 3, 0, 2])
    sorted_df = df.sort_values(by=['a', 'b'])
    tm.assert_frame_equal(sorted_df, expected)