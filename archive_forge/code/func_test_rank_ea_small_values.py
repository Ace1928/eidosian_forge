from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_rank_ea_small_values(self):
    ser = Series([5.4954145e+29, -9.791984e-21, 9.3715776e-26, NA, 1.8790257e-28], dtype='Float64')
    result = ser.rank(method='min')
    expected = Series([4, 1, 3, np.nan, 2])
    tm.assert_series_equal(result, expected)