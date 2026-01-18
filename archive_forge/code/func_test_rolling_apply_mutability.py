import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_rolling_apply_mutability(self):
    df = DataFrame({'A': ['foo'] * 3 + ['bar'] * 3, 'B': [1] * 6})
    g = df.groupby('A')
    mi = MultiIndex.from_tuples([('bar', 3), ('bar', 4), ('bar', 5), ('foo', 0), ('foo', 1), ('foo', 2)])
    mi.names = ['A', None]
    expected = DataFrame([np.nan, 2.0, 2.0] * 2, columns=['B'], index=mi)
    result = g.rolling(window=2).sum()
    tm.assert_frame_equal(result, expected)
    g.sum()
    result = g.rolling(window=2).sum()
    tm.assert_frame_equal(result, expected)