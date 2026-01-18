from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_multiindex_col(self):
    vals = np.random.default_rng(2).standard_normal((3, 3)).astype(object)
    idx = ['x', 'y', 'z']
    full = np.hstack(([[x] for x in idx], vals))
    df = DataFrame(vals, Index(idx, name='a'), columns=[['b', 'b', 'c'], ['mean', 'median', 'mean']])
    rs = df.reset_index()
    xp = DataFrame(full, columns=[['a', 'b', 'b', 'c'], ['', 'mean', 'median', 'mean']])
    tm.assert_frame_equal(rs, xp)
    rs = df.reset_index(col_fill=None)
    xp = DataFrame(full, columns=[['a', 'b', 'b', 'c'], ['a', 'mean', 'median', 'mean']])
    tm.assert_frame_equal(rs, xp)
    rs = df.reset_index(col_level=1, col_fill='blah')
    xp = DataFrame(full, columns=[['blah', 'b', 'b', 'c'], ['a', 'mean', 'median', 'mean']])
    tm.assert_frame_equal(rs, xp)
    df = DataFrame(vals, MultiIndex.from_arrays([[0, 1, 2], ['x', 'y', 'z']], names=['d', 'a']), columns=[['b', 'b', 'c'], ['mean', 'median', 'mean']])
    rs = df.reset_index('a')
    xp = DataFrame(full, Index([0, 1, 2], name='d'), columns=[['a', 'b', 'b', 'c'], ['', 'mean', 'median', 'mean']])
    tm.assert_frame_equal(rs, xp)
    rs = df.reset_index('a', col_fill=None)
    xp = DataFrame(full, Index(range(3), name='d'), columns=[['a', 'b', 'b', 'c'], ['a', 'mean', 'median', 'mean']])
    tm.assert_frame_equal(rs, xp)
    rs = df.reset_index('a', col_fill='blah', col_level=1)
    xp = DataFrame(full, Index(range(3), name='d'), columns=[['blah', 'b', 'b', 'c'], ['a', 'mean', 'median', 'mean']])
    tm.assert_frame_equal(rs, xp)