import numpy as np
from pandas import (
import pandas._testing as tm
def test_value_counts_unique_datetimeindex2(self, tz_naive_fixture):
    tz = tz_naive_fixture
    idx = DatetimeIndex(['2013-01-01 09:00', '2013-01-01 09:00', '2013-01-01 09:00', '2013-01-01 08:00', '2013-01-01 08:00', NaT], tz=tz)
    self._check_value_counts_dropna(idx)