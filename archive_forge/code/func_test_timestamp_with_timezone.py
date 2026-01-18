import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_timestamp_with_timezone(self, unit):
    left = DatetimeIndex(['2020-01-01'], dtype=f'M8[{unit}, UTC]')
    right = DatetimeIndex(['2020-01-02'], dtype=f'M8[{unit}, UTC]')
    index = IntervalIndex.from_arrays(left, right)
    result = repr(index)
    expected = f"IntervalIndex([(2020-01-01 00:00:00+00:00, 2020-01-02 00:00:00+00:00]], dtype='interval[datetime64[{unit}, UTC], right]')"
    assert result == expected