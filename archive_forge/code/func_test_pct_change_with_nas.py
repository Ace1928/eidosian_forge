import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('periods, fill_method, limit, exp', [(1, 'ffill', None, [np.nan, np.nan, np.nan, 1, 1, 1.5, 0, 0]), (1, 'ffill', 1, [np.nan, np.nan, np.nan, 1, 1, 1.5, 0, np.nan]), (1, 'bfill', None, [np.nan, 0, 0, 1, 1, 1.5, np.nan, np.nan]), (1, 'bfill', 1, [np.nan, np.nan, 0, 1, 1, 1.5, np.nan, np.nan]), (-1, 'ffill', None, [np.nan, np.nan, -0.5, -0.5, -0.6, 0, 0, np.nan]), (-1, 'ffill', 1, [np.nan, np.nan, -0.5, -0.5, -0.6, 0, np.nan, np.nan]), (-1, 'bfill', None, [0, 0, -0.5, -0.5, -0.6, np.nan, np.nan, np.nan]), (-1, 'bfill', 1, [np.nan, 0, -0.5, -0.5, -0.6, np.nan, np.nan, np.nan])])
def test_pct_change_with_nas(self, periods, fill_method, limit, exp, frame_or_series):
    vals = [np.nan, np.nan, 1, 2, 4, 10, np.nan, np.nan]
    obj = frame_or_series(vals)
    msg = f"The 'fill_method' keyword being not None and the 'limit' keyword in {type(obj).__name__}.pct_change are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = obj.pct_change(periods=periods, fill_method=fill_method, limit=limit)
    tm.assert_equal(res, frame_or_series(exp))