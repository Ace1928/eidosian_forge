import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('f', ['corr', 'cov'])
def test_rolling_corr_cov_other_diff_size_as_groups(self, f, roll_frame):
    g = roll_frame.groupby('A')
    r = g.rolling(window=4)
    result = getattr(r, f)(roll_frame)

    def func(x):
        return getattr(x.rolling(4), f)(roll_frame)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = g.apply(func)
    expected['A'] = np.nan
    tm.assert_frame_equal(result, expected)