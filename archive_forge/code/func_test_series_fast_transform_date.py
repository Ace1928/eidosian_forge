import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_series_fast_transform_date():
    df = DataFrame({'grouping': [np.nan, 1, 1, 3], 'd': date_range('2014-1-1', '2014-1-4')})
    result = df.groupby('grouping')['d'].transform('first')
    dates = [pd.NaT, Timestamp('2014-1-2'), Timestamp('2014-1-2'), Timestamp('2014-1-4')]
    expected = Series(dates, name='d', dtype='M8[ns]')
    tm.assert_series_equal(result, expected)