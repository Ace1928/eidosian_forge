import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_values_loses_freq_of_underlying_index():
    idx = pd.DatetimeIndex(date_range('20200101', periods=3, freq='BME'))
    expected = idx.copy(deep=True)
    idx2 = Index([1, 2, 3])
    midx = MultiIndex(levels=[idx, idx2], codes=[[0, 1, 2], [0, 1, 2]])
    midx.values
    assert idx.freq is not None
    tm.assert_index_equal(idx, expected)