import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_values_with_na(self):
    arrays = [array([2, NA, 1], dtype='Int64'), array([1, 2, 3], dtype='Int64')]
    index = MultiIndex.from_arrays(arrays)
    index = index.sort_values()
    result = DataFrame(range(3), index=index)
    arrays = [array([1, 2, NA], dtype='Int64'), array([3, 1, 2], dtype='Int64')]
    index = MultiIndex.from_arrays(arrays)
    expected = DataFrame(range(3), index=index)
    tm.assert_frame_equal(result, expected)