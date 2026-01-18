import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_weighted_var_big_window_no_segfault(win_types, center):
    pytest.importorskip('scipy')
    x = Series(0)
    result = x.rolling(window=16, center=center, win_type=win_types).var()
    expected = Series(np.nan)
    tm.assert_series_equal(result, expected)