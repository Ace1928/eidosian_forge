import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_axis_mixed(self, interp_method, request, using_array_manager):
    interpolation, method = interp_method
    df = DataFrame({'A': [1, 2, 3], 'B': [2.0, 3.0, 4.0], 'C': pd.date_range('20130101', periods=3), 'D': ['foo', 'bar', 'baz']})
    result = df.quantile(0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method)
    expected = Series([1.5, 2.5, 3.5], name=0.5)
    if interpolation == 'nearest':
        expected -= 0.5
    if method == 'table' and using_array_manager:
        request.applymarker(pytest.mark.xfail(reason='Axis name incorrectly set.'))
    tm.assert_series_equal(result, expected)
    msg = "'<' not supported between instances of 'Timestamp' and 'float'"
    with pytest.raises(TypeError, match=msg):
        df.quantile(0.5, axis=1, numeric_only=False)