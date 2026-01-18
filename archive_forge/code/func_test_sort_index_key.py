import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_key(self):
    df = DataFrame(np.arange(6, dtype='int64'), index=list('aaBBca'))
    result = df.sort_index()
    expected = df.iloc[[2, 3, 0, 1, 5, 4]]
    tm.assert_frame_equal(result, expected)
    result = df.sort_index(key=lambda x: x.str.lower())
    expected = df.iloc[[0, 1, 5, 2, 3, 4]]
    tm.assert_frame_equal(result, expected)
    result = df.sort_index(key=lambda x: x.str.lower(), ascending=False)
    expected = df.iloc[[4, 2, 3, 0, 1, 5]]
    tm.assert_frame_equal(result, expected)