from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_time_field_bug():
    df = DataFrame({'a': 1, 'b': [datetime.now() for nn in range(10)]})

    def func_with_no_date(batch):
        return Series({'c': 2})

    def func_with_date(batch):
        return Series({'b': datetime(2015, 1, 1), 'c': 2})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        dfg_no_conversion = df.groupby(by=['a']).apply(func_with_no_date)
    dfg_no_conversion_expected = DataFrame({'c': 2}, index=[1])
    dfg_no_conversion_expected.index.name = 'a'
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        dfg_conversion = df.groupby(by=['a']).apply(func_with_date)
    dfg_conversion_expected = DataFrame({'b': pd.Timestamp(2015, 1, 1).as_unit('ns'), 'c': 2}, index=[1])
    dfg_conversion_expected.index.name = 'a'
    tm.assert_frame_equal(dfg_no_conversion, dfg_no_conversion_expected)
    tm.assert_frame_equal(dfg_conversion, dfg_conversion_expected)