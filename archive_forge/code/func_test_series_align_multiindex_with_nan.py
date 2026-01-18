import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
def test_series_align_multiindex_with_nan(self):
    mi1 = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
    mi2 = MultiIndex.from_arrays([[np.nan, 81.0], [np.nan, np.nan]])
    ser1 = Series([1, 2], index=mi1)
    ser2 = Series([1, 2], index=mi2)
    result1, result2 = ser1.align(ser2)
    mi = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
    expected1 = Series([1, 2], index=mi)
    expected2 = Series([2, 1], index=mi)
    tm.assert_series_equal(result1, expected1)
    tm.assert_series_equal(result2, expected2)