import numpy as np
from pandas import (
import pandas._testing as tm
def test_value_counts_unique_datetimeindex(self, tz_naive_fixture):
    tz = tz_naive_fixture
    orig = date_range('2011-01-01 09:00', freq='h', periods=10, tz=tz)
    self._check_value_counts_with_repeats(orig)