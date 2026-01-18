import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_datetime_to_numeric():
    df = DataFrame({'a': 1, 'b': date_range('2015-01-01', periods=2, freq='D')})
    result = df.groupby('a').b.transform(lambda x: x.dt.dayofweek - x.dt.dayofweek.mean())
    expected = Series([-0.5, 0.5], name='b')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'a': 1, 'b': date_range('2015-01-01', periods=2, freq='D')})
    result = df.groupby('a').b.transform(lambda x: x.dt.dayofweek - x.dt.dayofweek.min())
    expected = Series([0, 1], dtype=np.int32, name='b')
    tm.assert_series_equal(result, expected)