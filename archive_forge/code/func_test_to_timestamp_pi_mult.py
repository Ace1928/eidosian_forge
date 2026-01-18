from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_timestamp_pi_mult(self):
    idx = PeriodIndex(['2011-01', 'NaT', '2011-02'], freq='2M', name='idx')
    result = idx.to_timestamp()
    expected = DatetimeIndex(['2011-01-01', 'NaT', '2011-02-01'], dtype='M8[ns]', name='idx')
    tm.assert_index_equal(result, expected)
    result = idx.to_timestamp(how='E')
    expected = DatetimeIndex(['2011-02-28', 'NaT', '2011-03-31'], dtype='M8[ns]', name='idx')
    expected = expected + Timedelta(1, 'D') - Timedelta(1, 'ns')
    tm.assert_index_equal(result, expected)