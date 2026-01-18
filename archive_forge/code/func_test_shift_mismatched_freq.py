import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_mismatched_freq(self, frame_or_series):
    ts = frame_or_series(np.random.default_rng(2).standard_normal(5), index=date_range('1/1/2000', periods=5, freq='h'))
    result = ts.shift(1, freq='5min')
    exp_index = ts.index.shift(1, freq='5min')
    tm.assert_index_equal(result.index, exp_index)
    result = ts.shift(1, freq='4h')
    exp_index = ts.index + offsets.Hour(4)
    tm.assert_index_equal(result.index, exp_index)