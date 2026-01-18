from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
@pytest.mark.parametrize('method', [True, False])
def test_pivot_with_tz(self, method, unit):
    df = DataFrame({'dt1': pd.DatetimeIndex([datetime(2013, 1, 1, 9, 0), datetime(2013, 1, 2, 9, 0), datetime(2013, 1, 1, 9, 0), datetime(2013, 1, 2, 9, 0)], dtype=f'M8[{unit}, US/Pacific]'), 'dt2': pd.DatetimeIndex([datetime(2014, 1, 1, 9, 0), datetime(2014, 1, 1, 9, 0), datetime(2014, 1, 2, 9, 0), datetime(2014, 1, 2, 9, 0)], dtype=f'M8[{unit}, Asia/Tokyo]'), 'data1': np.arange(4, dtype='int64'), 'data2': np.arange(4, dtype='int64')})
    exp_col1 = Index(['data1', 'data1', 'data2', 'data2'])
    exp_col2 = pd.DatetimeIndex(['2014/01/01 09:00', '2014/01/02 09:00'] * 2, name='dt2', dtype=f'M8[{unit}, Asia/Tokyo]')
    exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
    exp_idx = pd.DatetimeIndex(['2013/01/01 09:00', '2013/01/02 09:00'], name='dt1', dtype=f'M8[{unit}, US/Pacific]')
    expected = DataFrame([[0, 2, 0, 2], [1, 3, 1, 3]], index=exp_idx, columns=exp_col)
    if method:
        pv = df.pivot(index='dt1', columns='dt2')
    else:
        pv = pd.pivot(df, index='dt1', columns='dt2')
    tm.assert_frame_equal(pv, expected)
    expected = DataFrame([[0, 2], [1, 3]], index=exp_idx, columns=exp_col2[:2])
    if method:
        pv = df.pivot(index='dt1', columns='dt2', values='data1')
    else:
        pv = pd.pivot(df, index='dt1', columns='dt2', values='data1')
    tm.assert_frame_equal(pv, expected)