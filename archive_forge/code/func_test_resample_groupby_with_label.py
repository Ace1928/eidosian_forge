from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_resample_groupby_with_label(unit):
    index = date_range('2000-01-01', freq='2D', periods=5, unit=unit)
    df = DataFrame(index=index, data={'col0': [0, 0, 1, 1, 2], 'col1': [1, 1, 1, 1, 1]})
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('col0').resample('1W', label='left').sum()
    mi = [np.array([0, 0, 1, 2], dtype=np.int64), np.array(['1999-12-26', '2000-01-02', '2000-01-02', '2000-01-02'], dtype=f'M8[{unit}]')]
    mindex = pd.MultiIndex.from_arrays(mi, names=['col0', None])
    expected = DataFrame(data={'col0': [0, 0, 2, 2], 'col1': [1, 1, 2, 1]}, index=mindex)
    tm.assert_frame_equal(result, expected)