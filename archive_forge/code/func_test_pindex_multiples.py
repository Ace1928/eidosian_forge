import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_pindex_multiples(self):
    expected = PeriodIndex(['2011-01', '2011-03', '2011-05', '2011-07', '2011-09', '2011-11'], freq='2M')
    pi = period_range(start='1/1/11', end='12/31/11', freq='2M')
    tm.assert_index_equal(pi, expected)
    assert pi.freq == offsets.MonthEnd(2)
    assert pi.freqstr == '2M'
    pi = period_range(start='1/1/11', periods=6, freq='2M')
    tm.assert_index_equal(pi, expected)
    assert pi.freq == offsets.MonthEnd(2)
    assert pi.freqstr == '2M'