import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('ignore_index', [True, False])
def test_respect_ignore_index(self, inplace, ignore_index):
    df = DataFrame({'a': [1, 2, 3]}, index=RangeIndex(4, -1, -2))
    result = df.sort_index(ascending=False, ignore_index=ignore_index, inplace=inplace)
    if inplace:
        result = df
    if ignore_index:
        expected = DataFrame({'a': [1, 2, 3]})
    else:
        expected = DataFrame({'a': [1, 2, 3]}, index=RangeIndex(4, -1, -2))
    tm.assert_frame_equal(result, expected)