from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_map_box_timestamps():
    ser = Series(date_range('1/1/2000', periods=3))

    def func(x):
        return (x.hour, x.day, x.month)
    result = ser.map(func)
    expected = Series([(0, 1, 1), (0, 2, 1), (0, 3, 1)])
    tm.assert_series_equal(result, expected)