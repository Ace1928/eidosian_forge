import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_nat(self, interp_method, request, using_array_manager, unit):
    interpolation, method = interp_method
    if method == 'table' and using_array_manager:
        request.applymarker(pytest.mark.xfail(reason='Axis name incorrectly set.'))
    df = DataFrame({'a': [pd.NaT, pd.NaT, pd.NaT]}, dtype=f'M8[{unit}]')
    res = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
    exp = Series([pd.NaT], index=['a'], name=0.5, dtype=f'M8[{unit}]')
    tm.assert_series_equal(res, exp)
    res = df.quantile([0.5], numeric_only=False, interpolation=interpolation, method=method)
    exp = DataFrame({'a': [pd.NaT]}, index=[0.5], dtype=f'M8[{unit}]')
    tm.assert_frame_equal(res, exp)
    df = DataFrame({'a': [Timestamp('2012-01-01'), Timestamp('2012-01-02'), Timestamp('2012-01-03')], 'b': [pd.NaT, pd.NaT, pd.NaT]}, dtype=f'M8[{unit}]')
    res = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
    exp = Series([Timestamp('2012-01-02'), pd.NaT], index=['a', 'b'], name=0.5, dtype=f'M8[{unit}]')
    tm.assert_series_equal(res, exp)
    res = df.quantile([0.5], numeric_only=False, interpolation=interpolation, method=method)
    exp = DataFrame([[Timestamp('2012-01-02'), pd.NaT]], index=[0.5], columns=['a', 'b'], dtype=f'M8[{unit}]')
    tm.assert_frame_equal(res, exp)