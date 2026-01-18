import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_not_yet_implemented
def test_shift_axis1_multiple_blocks_with_int_fill(self):
    rng = np.random.default_rng(2)
    df1 = DataFrame(rng.integers(1000, size=(5, 3), dtype=int))
    df2 = DataFrame(rng.integers(1000, size=(5, 2), dtype=int))
    df3 = pd.concat([df1.iloc[:4, 1:3], df2.iloc[:4, :]], axis=1)
    result = df3.shift(2, axis=1, fill_value=np.int_(0))
    assert len(df3._mgr.blocks) == 2
    expected = df3.take([-1, -1, 0, 1], axis=1)
    expected.iloc[:, :2] = np.int_(0)
    expected.columns = df3.columns
    tm.assert_frame_equal(result, expected)
    df3 = pd.concat([df1.iloc[:4, 1:3], df2.iloc[:4, :]], axis=1)
    result = df3.shift(-2, axis=1, fill_value=np.int_(0))
    assert len(df3._mgr.blocks) == 2
    expected = df3.take([2, 3, -1, -1], axis=1)
    expected.iloc[:, -2:] = np.int_(0)
    expected.columns = df3.columns
    tm.assert_frame_equal(result, expected)