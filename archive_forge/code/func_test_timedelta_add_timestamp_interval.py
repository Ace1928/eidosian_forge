from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('klass', [timedelta, np.timedelta64, Timedelta])
def test_timedelta_add_timestamp_interval(self, klass):
    delta = klass(0)
    expected = Interval(Timestamp('2020-01-01'), Timestamp('2020-02-01'))
    result = delta + expected
    assert result == expected
    result = expected + delta
    assert result == expected