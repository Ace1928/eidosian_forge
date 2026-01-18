from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_where_tz(self):
    i = date_range('20130101', periods=3, tz='US/Eastern')
    result = i.where(notna(i))
    expected = i
    tm.assert_index_equal(result, expected)
    i2 = i.copy()
    i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
    result = i.where(notna(i2))
    expected = i2
    tm.assert_index_equal(result, expected)