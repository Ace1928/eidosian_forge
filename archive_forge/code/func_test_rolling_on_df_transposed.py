from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_on_df_transposed():
    df = DataFrame({'A': [1, None], 'B': [4, 5], 'C': [7, 8]})
    expected = DataFrame({'A': [1.0, np.nan], 'B': [5.0, 5.0], 'C': [11.0, 13.0]})
    msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(min_periods=1, window=2, axis=1).sum()
    tm.assert_frame_equal(result, expected)
    result = df.T.rolling(min_periods=1, window=2).sum().T
    tm.assert_frame_equal(result, expected)