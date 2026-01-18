import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_null_group_str_reducer(request, dropna, reduction_func):
    if reduction_func == 'corrwith':
        msg = 'incorrectly raises'
        request.applymarker(pytest.mark.xfail(reason=msg))
    index = [1, 2, 3, 4]
    df = DataFrame({'A': [1, 1, np.nan, np.nan], 'B': [1, 2, 2, 3]}, index=index)
    gb = df.groupby('A', dropna=dropna)
    args = get_groupby_method_args(reduction_func, df)
    if reduction_func == 'first':
        expected = DataFrame({'B': [1, 1, 2, 2]}, index=index)
    elif reduction_func == 'last':
        expected = DataFrame({'B': [2, 2, 3, 3]}, index=index)
    elif reduction_func == 'nth':
        expected = DataFrame({'B': [1, 1, 2, 2]}, index=index)
    elif reduction_func == 'size':
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == 'corrwith':
        expected = DataFrame({'B': [1.0, 1.0, 1.0, 1.0]}, index=index)
    else:
        expected_gb = df.groupby('A', dropna=False)
        buffer = []
        for idx, group in expected_gb:
            res = getattr(group['B'], reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer).to_frame('B')
    if dropna:
        dtype = object if reduction_func in ('any', 'all') else float
        expected = expected.astype(dtype)
        if expected.ndim == 2:
            expected.iloc[[2, 3], 0] = np.nan
        else:
            expected.iloc[[2, 3]] = np.nan
    result = gb.transform(reduction_func, *args)
    tm.assert_equal(result, expected)