import numpy as np
from pandas import (
import pandas._testing as tm
def test_value_counts_unique_periodindex2(self):
    idx = PeriodIndex(['2013-01-01 09:00', '2013-01-01 09:00', '2013-01-01 09:00', '2013-01-01 08:00', '2013-01-01 08:00', NaT], freq='h')
    self._check_value_counts_dropna(idx)