from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_unstack_categorical():
    df = DataFrame({'a': range(10), 'medium': ['A', 'B'] * 5, 'artist': list('XYXXY') * 2})
    df['medium'] = df['medium'].astype('category')
    gcat = df.groupby(['artist', 'medium'], observed=False)['a'].count().unstack()
    result = gcat.describe()
    exp_columns = CategoricalIndex(['A', 'B'], ordered=False, name='medium')
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat['A'] + gcat['B']
    expected = Series([6, 4], index=Index(['X', 'Y'], name='artist'))
    tm.assert_series_equal(result, expected)