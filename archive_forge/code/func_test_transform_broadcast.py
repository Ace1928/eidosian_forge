import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_broadcast(tsframe, ts):
    grouped = ts.groupby(lambda x: x.month)
    msg = 'using SeriesGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grouped.transform(np.mean)
    tm.assert_index_equal(result.index, ts.index)
    for _, gp in grouped:
        assert_fp_equal(result.reindex(gp.index), gp.mean())
    grouped = tsframe.groupby(lambda x: x.month)
    msg = 'using DataFrameGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grouped.transform(np.mean)
    tm.assert_index_equal(result.index, tsframe.index)
    for _, gp in grouped:
        agged = gp.mean(axis=0)
        res = result.reindex(gp.index)
        for col in tsframe:
            assert_fp_equal(res[col], agged[col])
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped = tsframe.groupby({'A': 0, 'B': 0, 'C': 1, 'D': 1}, axis=1)
    msg = 'using DataFrameGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grouped.transform(np.mean)
    tm.assert_index_equal(result.index, tsframe.index)
    tm.assert_index_equal(result.columns, tsframe.columns)
    for _, gp in grouped:
        agged = gp.mean(1)
        res = result.reindex(columns=gp.columns)
        for idx in gp.index:
            assert_fp_equal(res.xs(idx), agged[idx])