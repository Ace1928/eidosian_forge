import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_interpolate_method_values(self):
    rng = date_range('1/1/2000', '1/20/2000', freq='D')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts[::2] = np.nan
    result = ts.interpolate(method='values')
    exp = ts.interpolate()
    tm.assert_series_equal(result, exp)