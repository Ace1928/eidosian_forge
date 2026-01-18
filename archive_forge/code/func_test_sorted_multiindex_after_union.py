import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sorted_multiindex_after_union(self):
    midx = MultiIndex.from_product([pd.date_range('20110101', periods=2), Index(['a', 'b'])])
    ser1 = Series(1, index=midx)
    ser2 = Series(1, index=midx[:2])
    df = pd.concat([ser1, ser2], axis=1)
    expected = df.copy()
    result = df.loc['2011-01-01':'2011-01-02']
    tm.assert_frame_equal(result, expected)
    df = DataFrame({0: ser1, 1: ser2})
    result = df.loc['2011-01-01':'2011-01-02']
    tm.assert_frame_equal(result, expected)
    df = pd.concat([ser1, ser2.reindex(ser1.index)], axis=1)
    result = df.loc['2011-01-01':'2011-01-02']
    tm.assert_frame_equal(result, expected)