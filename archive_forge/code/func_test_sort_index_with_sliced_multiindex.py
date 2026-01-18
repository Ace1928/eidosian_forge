import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_with_sliced_multiindex():
    mi = MultiIndex.from_tuples([('a', '10'), ('a', '18'), ('a', '25'), ('b', '16'), ('b', '26'), ('a', '45'), ('b', '28'), ('a', '5'), ('a', '50'), ('a', '51'), ('b', '4')], names=['group', 'str'])
    df = DataFrame({'x': range(len(mi))}, index=mi)
    result = df.iloc[0:6].sort_index()
    expected = DataFrame({'x': [0, 1, 2, 5, 3, 4]}, index=MultiIndex.from_tuples([('a', '10'), ('a', '18'), ('a', '25'), ('a', '45'), ('b', '16'), ('b', '26')], names=['group', 'str']))
    tm.assert_frame_equal(result, expected)