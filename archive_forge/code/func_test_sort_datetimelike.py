from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('ordered', [True, False])
def test_sort_datetimelike(sort, ordered):
    df = DataFrame({'dt': [datetime(2011, 7, 1), datetime(2011, 7, 1), datetime(2011, 2, 1), datetime(2011, 5, 1), datetime(2011, 2, 1), datetime(2011, 1, 1), datetime(2011, 5, 1)], 'foo': [10, 8, 5, 6, 4, 1, 7], 'bar': [10, 20, 30, 40, 50, 60, 70]}, columns=['dt', 'foo', 'bar'])
    df['dt'] = Categorical(df['dt'], ordered=ordered)
    if sort:
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = [datetime(2011, 1, 1), datetime(2011, 2, 1), datetime(2011, 5, 1), datetime(2011, 7, 1)]
    else:
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = [datetime(2011, 7, 1), datetime(2011, 2, 1), datetime(2011, 5, 1), datetime(2011, 1, 1)]
    expected = DataFrame(data_values, columns=['foo', 'bar'], index=CategoricalIndex(index_values, name='dt', ordered=ordered))
    result = df.groupby('dt', sort=sort, observed=False).first()
    tm.assert_frame_equal(result, expected)