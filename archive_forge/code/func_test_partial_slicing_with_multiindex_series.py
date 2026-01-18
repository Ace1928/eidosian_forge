from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slicing_with_multiindex_series(self):
    ser = Series(range(250), index=MultiIndex.from_product([date_range('2000-1-1', periods=50), range(5)]))
    s2 = ser[:-1].copy()
    expected = s2['2000-1-4']
    result = s2[Timestamp('2000-1-4')]
    tm.assert_series_equal(result, expected)
    result = ser[Timestamp('2000-1-4')]
    expected = ser['2000-1-4']
    tm.assert_series_equal(result, expected)
    df2 = DataFrame(ser)
    expected = df2.xs('2000-1-4')
    result = df2.loc[Timestamp('2000-1-4')]
    tm.assert_frame_equal(result, expected)