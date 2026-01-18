import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('freq_depr, start, end, expected_values, expected_freq', [('3.5S', '05:03:01', '05:03:10', ['0 days 05:03:01', '0 days 05:03:04.500000', '0 days 05:03:08'], '3500ms'), ('2.5T', '5 hours', '5 hours 8 minutes', ['0 days 05:00:00', '0 days 05:02:30', '0 days 05:05:00', '0 days 05:07:30'], '150s')])
def test_timedelta_range_deprecated_freq(self, freq_depr, start, end, expected_values, expected_freq):
    msg = f"'{freq_depr[-1]}' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = timedelta_range(start=start, end=end, freq=freq_depr)
    expected = TimedeltaIndex(expected_values, dtype='timedelta64[ns]', freq=expected_freq)
    tm.assert_index_equal(result, expected)