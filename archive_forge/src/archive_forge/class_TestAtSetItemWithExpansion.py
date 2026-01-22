from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
class TestAtSetItemWithExpansion:

    def test_at_setitem_expansion_series_dt64tz_value(self, tz_naive_fixture):
        ts = Timestamp('2017-08-05 00:00:00+0100', tz=tz_naive_fixture)
        result = Series(ts)
        result.at[1] = ts
        expected = Series([ts, ts])
        tm.assert_series_equal(result, expected)