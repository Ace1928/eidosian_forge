import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('ordered', [True, False])
def test_sort_values_key_casts_to_categorical(self, ordered):
    categories = ['c', 'b', 'a']
    df = DataFrame({'x': [1, 1, 1], 'y': ['a', 'b', 'c']})

    def sorter(key):
        if key.name == 'y':
            return pd.Series(Categorical(key, categories=categories, ordered=ordered))
        return key
    result = df.sort_values(by=['x', 'y'], key=sorter)
    expected = DataFrame({'x': [1, 1, 1], 'y': ['c', 'b', 'a']}, index=pd.Index([2, 1, 0]))
    tm.assert_frame_equal(result, expected)