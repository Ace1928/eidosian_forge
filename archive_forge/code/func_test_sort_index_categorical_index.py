import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_categorical_index(self):
    df = DataFrame({'A': np.arange(6, dtype='int64'), 'B': Series(list('aabbca')).astype(CategoricalDtype(list('cab')))}).set_index('B')
    result = df.sort_index()
    expected = df.iloc[[4, 0, 1, 5, 2, 3]]
    tm.assert_frame_equal(result, expected)
    result = df.sort_index(ascending=False)
    expected = df.iloc[[2, 3, 0, 1, 5, 4]]
    tm.assert_frame_equal(result, expected)