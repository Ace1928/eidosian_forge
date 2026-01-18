import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('op_name', ['agg', 'apply'])
def test_with_nested_series(datetime_series, op_name):
    msg = 'cannot aggregate'
    warning = FutureWarning if op_name == 'agg' else None
    with tm.assert_produces_warning(warning, match=msg):
        result = getattr(datetime_series, op_name)(lambda x: Series([x, x ** 2], index=['x', 'x^2']))
    expected = DataFrame({'x': datetime_series, 'x^2': datetime_series ** 2})
    tm.assert_frame_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = datetime_series.agg(lambda x: Series([x, x ** 2], index=['x', 'x^2']))
    tm.assert_frame_equal(result, expected)