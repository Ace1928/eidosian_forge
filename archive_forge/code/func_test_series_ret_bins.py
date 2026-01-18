import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_series_ret_bins():
    ser = Series(np.arange(4))
    result, bins = cut(ser, 2, retbins=True)
    expected = Series(IntervalIndex.from_breaks([-0.003, 1.5, 3], closed='right').repeat(2)).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)