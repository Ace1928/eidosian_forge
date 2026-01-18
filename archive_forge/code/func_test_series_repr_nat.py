from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_repr_nat(self):
    series = Series([0, 1000, 2000, pd.NaT._value], dtype='M8[ns]')
    result = repr(series)
    expected = '0   1970-01-01 00:00:00.000000\n1   1970-01-01 00:00:00.000001\n2   1970-01-01 00:00:00.000002\n3                          NaT\ndtype: datetime64[ns]'
    assert result == expected