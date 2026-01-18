import pytest
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import TimedeltaIndex
import pandas._testing as tm
def test_tdi_shift_int(self):
    tdi = pd.to_timedelta(range(5), unit='d')
    trange = tdi._with_freq('infer') + pd.offsets.Hour(1)
    result = trange.shift(1)
    expected = TimedeltaIndex(['1 days 01:00:00', '2 days 01:00:00', '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00'], freq='D')
    tm.assert_index_equal(result, expected)