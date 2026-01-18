import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_round_with_duplicate_columns(self):
    df = DataFrame(np.random.default_rng(2).random([3, 3]), columns=['A', 'B', 'C'], index=['first', 'second', 'third'])
    dfs = pd.concat((df, df), axis=1)
    rounded = dfs.round()
    tm.assert_index_equal(rounded.index, dfs.index)
    decimals = Series([1, 0, 2], index=['A', 'B', 'A'])
    msg = 'Index of decimals must be unique'
    with pytest.raises(ValueError, match=msg):
        df.round(decimals)