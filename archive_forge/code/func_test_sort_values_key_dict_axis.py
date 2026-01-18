import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_key_dict_axis(self):
    df = DataFrame({0: ['Hello', 0], 1: ['goodbye', 1]})
    result = df.sort_values(0, key=lambda col: col.str.lower(), axis=1)
    expected = df.loc[:, ::-1]
    tm.assert_frame_equal(result, expected)
    result = df.sort_values(1, key=lambda col: -col, axis=1)
    expected = df.loc[:, ::-1]
    tm.assert_frame_equal(result, expected)