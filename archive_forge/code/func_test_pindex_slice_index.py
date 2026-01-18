import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_pindex_slice_index(self):
    pi = period_range(start='1/1/10', end='12/31/12', freq='M')
    s = Series(np.random.default_rng(2).random(len(pi)), index=pi)
    res = s['2010']
    exp = s[0:12]
    tm.assert_series_equal(res, exp)
    res = s['2011']
    exp = s[12:24]
    tm.assert_series_equal(res, exp)