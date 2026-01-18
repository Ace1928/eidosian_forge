from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_clipping_with_na_values(self, any_numeric_ea_dtype, nulls_fixture):
    if nulls_fixture is pd.NaT:
        pytest.skip('See test_constructor_mismatched_null_nullable_dtype')
    ser = Series([nulls_fixture, 1.0, 3.0], dtype=any_numeric_ea_dtype)
    s_clipped_upper = ser.clip(upper=2.0)
    s_clipped_lower = ser.clip(lower=2.0)
    expected_upper = Series([nulls_fixture, 1.0, 2.0], dtype=any_numeric_ea_dtype)
    expected_lower = Series([nulls_fixture, 2.0, 3.0], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(s_clipped_upper, expected_upper)
    tm.assert_series_equal(s_clipped_lower, expected_lower)