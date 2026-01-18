from datetime import timezone
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_tz_localize_axis1(self):
    rng = date_range('1/1/2011', periods=100, freq='h')
    df = DataFrame({'a': 1}, index=rng)
    df = df.T
    result = df.tz_localize('utc', axis=1)
    assert result.columns.tz is timezone.utc
    expected = DataFrame({'a': 1}, rng.tz_localize('UTC'))
    tm.assert_frame_equal(result, expected.T)