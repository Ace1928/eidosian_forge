import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('selection', ('b', ['b'], ['b', 'c']))
@pytest.mark.parametrize('dropna', ['any', 'all', None])
def test_nth_after_selection(selection, dropna):
    df = DataFrame({'a': [1, 1, 2], 'b': [np.nan, 3, 4], 'c': [5, 6, 7]})
    gb = df.groupby('a')[selection]
    result = gb.nth(0, dropna=dropna)
    if dropna == 'any' or (dropna == 'all' and selection != ['b', 'c']):
        locs = [1, 2]
    else:
        locs = [0, 2]
    expected = df.loc[locs, selection]
    tm.assert_equal(result, expected)