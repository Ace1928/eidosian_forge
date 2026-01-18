import numpy as np
from pandas import (
import pandas._testing as tm
def test_value_counts_unique_timedeltaindex(self):
    orig = timedelta_range('1 days 09:00:00', freq='h', periods=10)
    self._check_value_counts_with_repeats(orig)