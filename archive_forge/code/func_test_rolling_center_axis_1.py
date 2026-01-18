import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_rolling_center_axis_1():
    pytest.importorskip('scipy')
    df = DataFrame({'a': [1, 1, 0, 0, 0, 1], 'b': [1, 0, 0, 1, 0, 0], 'c': [1, 0, 0, 1, 0, 1]})
    msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(window=3, axis=1, win_type='boxcar', center=True).sum()
    expected = DataFrame({'a': [np.nan] * 6, 'b': [3.0, 1.0, 0.0, 2.0, 0.0, 2.0], 'c': [np.nan] * 6})
    tm.assert_frame_equal(result, expected, check_dtype=True)