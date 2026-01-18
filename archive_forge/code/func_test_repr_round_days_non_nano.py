import pytest
import pandas as pd
from pandas import (
def test_repr_round_days_non_nano(self):
    tdi = TimedeltaIndex(['1 days'], freq='D').as_unit('s')
    result = repr(tdi)
    expected = "TimedeltaIndex(['1 days'], dtype='timedelta64[s]', freq='D')"
    assert result == expected
    result2 = repr(Series(tdi))
    expected2 = '0   1 days\ndtype: timedelta64[s]'
    assert result2 == expected2