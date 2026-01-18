import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_level2(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    df = frame.copy()
    df.index = np.arange(len(df))
    a_sorted = frame['A'].sort_index(level=0)
    assert a_sorted.index.names == frame.index.names
    rs = frame.copy()
    return_value = rs.sort_index(level=0, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(rs, frame.sort_index(level=0))