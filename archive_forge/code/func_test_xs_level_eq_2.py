import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_level_eq_2(self):
    arr = np.random.default_rng(2).standard_normal((3, 5))
    index = MultiIndex(levels=[['a', 'p', 'x'], ['b', 'q', 'y'], ['c', 'r', 'z']], codes=[[2, 0, 1], [2, 0, 1], [2, 0, 1]])
    df = DataFrame(arr, index=index)
    expected = DataFrame(arr[1:2], index=[['a'], ['b']])
    result = df.xs('c', level=2)
    tm.assert_frame_equal(result, expected)