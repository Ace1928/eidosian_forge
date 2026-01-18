from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_rank_empty_group():
    column = 'A'
    df = DataFrame({'A': [0, 1, 0], 'B': [1.0, np.nan, 2.0]})
    result = df.groupby(column).B.rank(pct=True)
    expected = Series([0.5, np.nan, 1.0], name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby(column).rank(pct=True)
    expected = DataFrame({'B': [0.5, np.nan, 1.0]})
    tm.assert_frame_equal(result, expected)