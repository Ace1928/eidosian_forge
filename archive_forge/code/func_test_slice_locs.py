from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_slice_locs(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=50, freq='B'))
    stacked = df.stack(future_stack=True)
    idx = stacked.index
    slob = slice(*idx.slice_locs(df.index[5], df.index[15]))
    sliced = stacked[slob]
    expected = df[5:16].stack(future_stack=True)
    tm.assert_almost_equal(sliced.values, expected.values)
    slob = slice(*idx.slice_locs(df.index[5] + timedelta(seconds=30), df.index[15] - timedelta(seconds=30)))
    sliced = stacked[slob]
    expected = df[6:15].stack(future_stack=True)
    tm.assert_almost_equal(sliced.values, expected.values)