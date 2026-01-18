import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dropna', [None, 'any', 'all'])
def test_nth_nan_in_grouper(dropna):
    df = DataFrame({'a': [np.nan, 'a', np.nan, 'b', np.nan], 'b': [0, 2, 4, 6, 8], 'c': [1, 3, 5, 7, 9]})
    result = df.groupby('a').nth(0, dropna=dropna)
    expected = df.iloc[[1, 3]]
    tm.assert_frame_equal(result, expected)