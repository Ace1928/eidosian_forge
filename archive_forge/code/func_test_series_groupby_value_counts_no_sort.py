from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_series_groupby_value_counts_no_sort():
    df = DataFrame({'gender': ['male', 'male', 'female', 'male', 'female', 'male'], 'education': ['low', 'medium', 'high', 'low', 'high', 'low'], 'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']})
    gb = df.groupby(['country', 'gender'], sort=False)['education']
    result = gb.value_counts(sort=False)
    index = MultiIndex(levels=[['US', 'FR'], ['male', 'female'], ['low', 'medium', 'high']], codes=[[0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 1, 2, 0, 2]], names=['country', 'gender', 'education'])
    expected = Series([1, 1, 1, 2, 1], index=index, name='count')
    tm.assert_series_equal(result, expected)