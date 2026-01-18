import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', [0, -1])
def test_nth_with_na_object(index, nulls_fixture):
    df = DataFrame({'a': [1, 1, 2, 2], 'b': [1, 2, 3, nulls_fixture]})
    groups = df.groupby('a')
    result = groups.nth(index)
    expected = df.iloc[[0, 2]] if index == 0 else df.iloc[[1, 3]]
    tm.assert_frame_equal(result, expected)