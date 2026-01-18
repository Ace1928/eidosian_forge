import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_timedelta_cut_roundtrip():
    ser = Series(timedelta_range('1day', periods=3))
    result, result_bins = cut(ser, 2, retbins=True)
    expected = cut(ser, result_bins)
    tm.assert_series_equal(result, expected)
    expected_bins = TimedeltaIndex(['0 days 23:57:07.200000', '2 days 00:00:00', '3 days 00:00:00'])
    tm.assert_index_equal(result_bins, expected_bins)