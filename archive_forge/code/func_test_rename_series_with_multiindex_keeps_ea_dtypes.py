from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_series_with_multiindex_keeps_ea_dtypes(self):
    arrays = [Index([1, 2, 3], dtype='Int64').astype('category'), Index([1, 2, 3], dtype='Int64')]
    mi = MultiIndex.from_arrays(arrays, names=['A', 'B'])
    ser = Series(1, index=mi)
    result = ser.rename({1: 4}, level=1)
    arrays_expected = [Index([1, 2, 3], dtype='Int64').astype('category'), Index([4, 2, 3], dtype='Int64')]
    mi_expected = MultiIndex.from_arrays(arrays_expected, names=['A', 'B'])
    expected = Series(1, index=mi_expected)
    tm.assert_series_equal(result, expected)