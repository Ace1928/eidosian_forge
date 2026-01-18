import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_pct_change_numeric(self):
    pnl = DataFrame([np.arange(0, 40, 10), np.arange(0, 40, 10), np.arange(0, 40, 10)]).astype(np.float64)
    pnl.iat[1, 0] = np.nan
    pnl.iat[1, 1] = np.nan
    pnl.iat[2, 3] = 60
    msg = "The 'fill_method' keyword being not None and the 'limit' keyword in DataFrame.pct_change are deprecated"
    for axis in range(2):
        expected = pnl.ffill(axis=axis) / pnl.ffill(axis=axis).shift(axis=axis) - 1
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = pnl.pct_change(axis=axis, fill_method='pad')
        tm.assert_frame_equal(result, expected)