from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('consolidate', [True, False])
def test_resample_groupby_agg_object_dtype_all_nan(consolidate):
    dates = date_range('2020-01-01', periods=15, freq='D')
    df1 = DataFrame({'key': 'A', 'date': dates, 'col1': range(15), 'col_object': 'val'})
    df2 = DataFrame({'key': 'B', 'date': dates, 'col1': range(15)})
    df = pd.concat([df1, df2], ignore_index=True)
    if consolidate:
        df = df._consolidate()
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby(['key']).resample('W', on='date').min()
    idx = pd.MultiIndex.from_arrays([['A'] * 3 + ['B'] * 3, pd.to_datetime(['2020-01-05', '2020-01-12', '2020-01-19'] * 2).as_unit('ns')], names=['key', 'date'])
    expected = DataFrame({'key': ['A'] * 3 + ['B'] * 3, 'col1': [0, 5, 12] * 2, 'col_object': ['val'] * 3 + [np.nan] * 3}, index=idx)
    tm.assert_frame_equal(result, expected)