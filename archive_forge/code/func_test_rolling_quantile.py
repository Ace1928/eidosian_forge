import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('interpolation', ['linear', 'lower', 'higher', 'midpoint', 'nearest'])
def test_rolling_quantile(self, interpolation, roll_frame):
    g = roll_frame.groupby('A', group_keys=False)
    r = g.rolling(window=4)
    result = r.quantile(0.4, interpolation=interpolation)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = g.apply(lambda x: x.rolling(4).quantile(0.4, interpolation=interpolation))
    expected = expected.drop('A', axis=1)
    expected_index = MultiIndex.from_arrays([roll_frame['A'], range(40)])
    expected.index = expected_index
    tm.assert_frame_equal(result, expected)