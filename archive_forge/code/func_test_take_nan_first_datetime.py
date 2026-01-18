from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_take_nan_first_datetime(self):
    index = DatetimeIndex([pd.NaT, Timestamp('20130101'), Timestamp('20130102')])
    result = index.take([-1, 0, 1])
    expected = DatetimeIndex([index[-1], index[0], index[1]])
    tm.assert_index_equal(result, expected)