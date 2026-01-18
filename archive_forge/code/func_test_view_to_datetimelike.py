import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_view_to_datetimelike():
    idx = Index([1, 2, 3])
    res = idx.view('m8[s]')
    expected = pd.TimedeltaIndex(idx.values.view('m8[s]'))
    tm.assert_index_equal(res, expected)
    res2 = idx.view('m8[D]')
    expected2 = idx.values.view('m8[D]')
    tm.assert_numpy_array_equal(res2, expected2)
    res3 = idx.view('M8[h]')
    expected3 = idx.values.view('M8[h]')
    tm.assert_numpy_array_equal(res3, expected3)