import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_empty_no_rows_ints(self, interp_method):
    interpolation, method = interp_method
    df = DataFrame(columns=['a', 'b'], dtype='int64')
    res = df.quantile(0.5, interpolation=interpolation, method=method)
    exp = Series([np.nan, np.nan], index=['a', 'b'], name=0.5)
    tm.assert_series_equal(res, exp)