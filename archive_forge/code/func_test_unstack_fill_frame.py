from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_fill_frame(self):
    rows = [[1, 2], [3, 4], [5, 6], [7, 8]]
    df = DataFrame(rows, columns=list('AB'), dtype=np.int32)
    df.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
    result = df.unstack(fill_value=-1)
    rows = [[1, 3, 2, 4], [-1, 5, -1, 6], [7, -1, 8, -1]]
    expected = DataFrame(rows, index=list('xyz'), dtype=np.int32)
    expected.columns = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')])
    tm.assert_frame_equal(result, expected)
    df['A'] = df['A'].astype(np.int16)
    df['B'] = df['B'].astype(np.float64)
    result = df.unstack(fill_value=-1)
    expected['A'] = expected['A'].astype(np.int16)
    expected['B'] = expected['B'].astype(np.float64)
    tm.assert_frame_equal(result, expected)
    result = df.unstack(fill_value=0.5)
    rows = [[1, 3, 2, 4], [0.5, 5, 0.5, 6], [7, 0.5, 8, 0.5]]
    expected = DataFrame(rows, index=list('xyz'), dtype=float)
    expected.columns = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')])
    tm.assert_frame_equal(result, expected)