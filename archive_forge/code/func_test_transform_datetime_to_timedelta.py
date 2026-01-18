import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_datetime_to_timedelta():
    df = DataFrame({'A': Timestamp('20130101'), 'B': np.arange(5)})
    expected = Series(Timestamp('20130101') - Timestamp('20130101'), index=range(5), name='A')
    base_time = df['A'][0]
    result = df.groupby('A')['A'].transform(lambda x: x.max() - x.min() + base_time) - base_time
    tm.assert_series_equal(result, expected)
    result = df.groupby('A')['A'].transform(lambda x: x.max() - x.min())
    tm.assert_series_equal(result, expected)