from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_where_doesnt_retain_freq(self):
    dti = date_range('20130101', periods=3, freq='D', name='idx')
    cond = [True, True, False]
    expected = DatetimeIndex([dti[0], dti[1], dti[0]], freq=None, name='idx')
    result = dti.where(cond, dti[::-1])
    tm.assert_index_equal(result, expected)