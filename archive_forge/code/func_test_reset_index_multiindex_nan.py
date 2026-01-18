from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_multiindex_nan(self):
    df = DataFrame({'A': ['a', 'b', 'c'], 'B': [0, 1, np.nan], 'C': np.random.default_rng(2).random(3)})
    rs = df.set_index(['A', 'B']).reset_index()
    tm.assert_frame_equal(rs, df)
    df = DataFrame({'A': [np.nan, 'b', 'c'], 'B': [0, 1, 2], 'C': np.random.default_rng(2).random(3)})
    rs = df.set_index(['A', 'B']).reset_index()
    tm.assert_frame_equal(rs, df)
    df = DataFrame({'A': ['a', 'b', 'c'], 'B': [0, 1, 2], 'C': [np.nan, 1.1, 2.2]})
    rs = df.set_index(['A', 'B']).reset_index()
    tm.assert_frame_equal(rs, df)
    df = DataFrame({'A': ['a', 'b', 'c'], 'B': [np.nan, np.nan, np.nan], 'C': np.random.default_rng(2).random(3)})
    rs = df.set_index(['A', 'B']).reset_index()
    tm.assert_frame_equal(rs, df)