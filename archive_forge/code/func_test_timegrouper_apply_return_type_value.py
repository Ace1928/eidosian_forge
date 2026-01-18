from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_timegrouper_apply_return_type_value(self):
    df = DataFrame({'date': ['10/10/2000', '11/10/2000'], 'value': [10, 13]})
    df_dt = df.copy()
    df_dt['date'] = pd.to_datetime(df_dt['date'])

    def sumfunc_value(x):
        return x.value.sum()
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = df.groupby(Grouper(key='date')).apply(sumfunc_value)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df_dt.groupby(Grouper(freq='ME', key='date')).apply(sumfunc_value)
    tm.assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True))