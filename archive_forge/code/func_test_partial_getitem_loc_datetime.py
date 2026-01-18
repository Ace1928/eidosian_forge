import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('indexer, exp_idx, exp_values', [(slice('2019-2', None), DatetimeIndex(['2019-02-01'], dtype='M8[ns]'), [2, 3]), (slice(None, '2019-2'), date_range('2019', periods=2, freq='MS'), [0, 1, 2, 3])])
def test_partial_getitem_loc_datetime(self, indexer, exp_idx, exp_values):
    date_idx = date_range('2019', periods=2, freq='MS')
    df = DataFrame(list(range(4)), index=MultiIndex.from_product([date_idx, [0, 1]], names=['x', 'y']))
    expected = DataFrame(exp_values, index=MultiIndex.from_product([exp_idx, [0, 1]], names=['x', 'y']))
    result = df[indexer]
    tm.assert_frame_equal(result, expected)
    result = df.loc[indexer]
    tm.assert_frame_equal(result, expected)
    result = df.loc(axis=0)[indexer]
    tm.assert_frame_equal(result, expected)
    result = df.loc[indexer, :]
    tm.assert_frame_equal(result, expected)
    df2 = df.swaplevel(0, 1).sort_index()
    expected = expected.swaplevel(0, 1).sort_index()
    result = df2.loc[:, indexer, :]
    tm.assert_frame_equal(result, expected)