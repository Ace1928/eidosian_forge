from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('value', ['test', to_datetime('2019-12-31'), to_timedelta('1 days 06:05:01.00003')])
def test_rolling_axis_1_non_numeric_dtypes(value):
    df = DataFrame({'a': [1, 2]})
    df['b'] = value
    msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(window=2, min_periods=1, axis=1).sum()
    expected = DataFrame({'a': [1.0, 2.0]})
    tm.assert_frame_equal(result, expected)