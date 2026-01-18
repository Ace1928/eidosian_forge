from functools import partial
import sys
import numpy as np
import pytest
import pandas._libs.window.aggregations as window_aggregations
from pandas import Series
import pandas._testing as tm
def test_rolling_aggregation_with_unused_elements(rolling_aggregation):
    minp, width = (0, 5)
    size = 2 * width + 5
    values = np.arange(1, size + 1, dtype=np.float64)
    values[width:width + 2] = sys.float_info.min
    values[width + 2] = np.nan
    values[width + 3:width + 5] = sys.float_info.max
    start = np.array([0, size - width], dtype=np.int64)
    end = np.array([width, size], dtype=np.int64)
    loc = np.array([j for i in range(len(start)) for j in range(start[i], end[i])], dtype=np.int32)
    result = Series(rolling_aggregation(values, start, end, minp))
    compact_values = np.array(values[loc], dtype=np.float64)
    compact_start = np.arange(0, len(start) * width, width, dtype=np.int64)
    compact_end = compact_start + width
    expected = Series(rolling_aggregation(compact_values, compact_start, compact_end, minp))
    assert np.isfinite(expected.values).all(), 'Not all expected values are finite'
    tm.assert_equal(expected, result)