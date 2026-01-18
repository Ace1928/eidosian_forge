import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_item_cache(self, using_array_manager, using_copy_on_write):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)), columns=['A', 'B', 'C'])
    df['D'] = df['A'] * 2
    ser = df['A']
    if not using_array_manager:
        assert len(df._mgr.blocks) == 2
    df.sort_values(by='A')
    if using_copy_on_write:
        ser.iloc[0] = 99
        assert df.iloc[0, 0] == df['A'][0]
        assert df.iloc[0, 0] != 99
    else:
        ser.values[0] = 99
        assert df.iloc[0, 0] == df['A'][0]
        assert df.iloc[0, 0] == 99