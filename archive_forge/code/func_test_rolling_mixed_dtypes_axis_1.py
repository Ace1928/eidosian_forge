from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize(('func', 'value'), [('sum', 2.0), ('max', 1.0), ('min', 1.0), ('mean', 1.0), ('median', 1.0)])
def test_rolling_mixed_dtypes_axis_1(func, value):
    df = DataFrame(1, index=[1, 2], columns=['a', 'b', 'c'])
    df['c'] = 1.0
    msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        roll = df.rolling(window=2, min_periods=1, axis=1)
    result = getattr(roll, func)()
    expected = DataFrame({'a': [1.0, 1.0], 'b': [value, value], 'c': [value, value]}, index=[1, 2])
    tm.assert_frame_equal(result, expected)