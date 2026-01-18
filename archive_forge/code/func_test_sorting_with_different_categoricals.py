from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_sorting_with_different_categoricals():
    df = DataFrame({'group': ['A'] * 6 + ['B'] * 6, 'dose': ['high', 'med', 'low'] * 4, 'outcomes': np.arange(12.0)})
    df.dose = Categorical(df.dose, categories=['low', 'med', 'high'], ordered=True)
    result = df.groupby('group')['dose'].value_counts()
    result = result.sort_index(level=0, sort_remaining=True)
    index = ['low', 'med', 'high', 'low', 'med', 'high']
    index = Categorical(index, categories=['low', 'med', 'high'], ordered=True)
    index = [['A', 'A', 'A', 'B', 'B', 'B'], CategoricalIndex(index)]
    index = MultiIndex.from_arrays(index, names=['group', 'dose'])
    expected = Series([2] * 6, index=index, name='count')
    tm.assert_series_equal(result, expected)