import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_axis_1(request, transformation_func):
    df = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}, index=['x', 'y'])
    args = get_groupby_method_args(transformation_func, df)
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby([0, 0, 1], axis=1)
    warn = FutureWarning if transformation_func == 'fillna' else None
    msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        result = gb.transform(transformation_func, *args)
    msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        expected = df.T.groupby([0, 0, 1]).transform(transformation_func, *args).T
    if transformation_func in ['diff', 'shift']:
        expected['b'] = expected['b'].astype('int64')
    tm.assert_equal(result, expected)