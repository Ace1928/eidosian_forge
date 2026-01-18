from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_resample_groupby_agg():
    df = DataFrame({'cat': ['cat_1', 'cat_1', 'cat_2', 'cat_1', 'cat_2', 'cat_1', 'cat_2', 'cat_1'], 'num': [5, 20, 22, 3, 4, 30, 10, 50], 'date': ['2019-2-1', '2018-02-03', '2020-3-11', '2019-2-2', '2019-2-2', '2018-12-4', '2020-3-11', '2020-12-12']})
    df['date'] = pd.to_datetime(df['date'])
    resampled = df.groupby('cat').resample('YE', on='date')
    expected = resampled[['num']].sum()
    result = resampled.agg({'num': 'sum'})
    tm.assert_frame_equal(result, expected)