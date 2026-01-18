import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_lambda_with_datetimetz():
    df = DataFrame({'time': [Timestamp('2010-07-15 03:14:45'), Timestamp('2010-11-19 18:47:06')], 'timezone': ['Etc/GMT+4', 'US/Eastern']})
    result = df.groupby(['timezone'])['time'].transform(lambda x: x.dt.tz_localize(x.name))
    expected = Series([Timestamp('2010-07-15 03:14:45', tz='Etc/GMT+4'), Timestamp('2010-11-19 18:47:06', tz='US/Eastern')], name='time')
    tm.assert_series_equal(result, expected)