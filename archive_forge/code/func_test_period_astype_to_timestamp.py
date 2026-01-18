import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_period_astype_to_timestamp(self, unit):
    pi = PeriodIndex(['2011-01', '2011-02', '2011-03'], freq='M')
    exp = DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'], tz='US/Eastern').as_unit(unit)
    res = pi.astype(f'datetime64[{unit}, US/Eastern]')
    tm.assert_index_equal(res, exp)
    assert res.freq == exp.freq