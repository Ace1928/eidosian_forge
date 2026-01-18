from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_numeric_dtypes():
    df = DataFrame(np.arange(40).reshape(4, 10), columns=list('abcdefghij')).astype({'a': 'float16', 'b': 'float32', 'c': 'float64', 'd': 'int8', 'e': 'int16', 'f': 'int32', 'g': 'uint8', 'h': 'uint16', 'i': 'uint32', 'j': 'uint64'})
    msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(window=2, min_periods=1, axis=1).min()
    expected = DataFrame({'a': range(0, 40, 10), 'b': range(0, 40, 10), 'c': range(1, 40, 10), 'd': range(2, 40, 10), 'e': range(3, 40, 10), 'f': range(4, 40, 10), 'g': range(5, 40, 10), 'h': range(6, 40, 10), 'i': range(7, 40, 10), 'j': range(8, 40, 10)}, dtype='float64')
    tm.assert_frame_equal(result, expected)