import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('msg_freq, freq', [('H', '19H12min'), ('T', '19h12T')])
def test_timedelta_range_H_T_deprecated(self, freq, msg_freq):
    msg = f"'{msg_freq}' is deprecated and will be removed in a future version."
    result = timedelta_range(start='0 days', end='4 days', periods=6)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = timedelta_range(start='0 days', end='4 days', freq=freq)
    tm.assert_index_equal(result, expected)