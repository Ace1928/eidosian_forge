from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('one_day_ten_secs', [Timedelta('1 day, 00:00:10'), Timedelta('1 days, 00:00:10'), timedelta(days=1, seconds=10), np.timedelta64(1, 'D') + np.timedelta64(10, 's'), offsets.Day() + offsets.Second(10)])
def test_td_add_sub_one_day_ten_seconds(self, one_day_ten_secs):
    base = Timestamp('20130102 09:01:12.123456')
    expected_add = Timestamp('20130103 09:01:22.123456')
    expected_sub = Timestamp('20130101 09:01:02.123456')
    result = base + one_day_ten_secs
    assert result == expected_add
    result = base - one_day_ten_secs
    assert result == expected_sub