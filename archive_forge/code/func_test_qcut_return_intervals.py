import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
def test_qcut_return_intervals():
    ser = Series([0, 1, 2, 3, 4, 5, 6, 7, 8])
    res = qcut(ser, [0, 0.333, 0.666, 1])
    exp_levels = np.array([Interval(-0.001, 2.664), Interval(2.664, 5.328), Interval(5.328, 8)])
    exp = Series(exp_levels.take([0, 0, 0, 1, 1, 1, 2, 2, 2])).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(res, exp)