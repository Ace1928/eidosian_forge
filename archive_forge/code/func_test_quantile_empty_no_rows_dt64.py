import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_empty_no_rows_dt64(self, interp_method):
    interpolation, method = interp_method
    df = DataFrame(columns=['a', 'b'], dtype='datetime64[ns]')
    res = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
    exp = Series([pd.NaT, pd.NaT], index=['a', 'b'], dtype='datetime64[ns]', name=0.5)
    tm.assert_series_equal(res, exp)
    df['a'] = df['a'].dt.tz_localize('US/Central')
    res = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
    exp = exp.astype(object)
    if interpolation == 'nearest':
        msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            exp = exp.fillna(np.nan, downcast=False)
    tm.assert_series_equal(res, exp)
    df['b'] = df['b'].dt.tz_localize('US/Central')
    res = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
    exp = exp.astype(df['b'].dtype)
    tm.assert_series_equal(res, exp)