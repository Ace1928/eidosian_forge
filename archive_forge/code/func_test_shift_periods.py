from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_periods(self, unit):
    idx = date_range(start=START, end=END, periods=3, unit=unit)
    tm.assert_index_equal(idx.shift(periods=0), idx)
    tm.assert_index_equal(idx.shift(0), idx)