from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('observed, index, data', [(True, MultiIndex.from_arrays([Index(['bar'] * 4 + ['foo'] * 4, dtype='category', name='A'), Index(['one', 'one', 'three', 'three', 'one', 'one', 'two', 'two'], dtype='category', name='B'), Index(['min', 'max'] * 4)]), [2, 2, 4, 4, 1, 1, 3, 3]), (False, MultiIndex.from_product([CategoricalIndex(['bar', 'foo'], ordered=False), CategoricalIndex(['one', 'three', 'two'], ordered=False), Index(['min', 'max'])], names=['A', 'B', None]), [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3]), (None, MultiIndex.from_product([CategoricalIndex(['bar', 'foo'], ordered=False), CategoricalIndex(['one', 'three', 'two'], ordered=False), Index(['min', 'max'])], names=['A', 'B', None]), [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3])])
def test_seriesgroupby_observed_apply_dict(df_cat, observed, index, data):
    expected = Series(data=data, index=index, name='C')
    result = df_cat.groupby(['A', 'B'], observed=observed)['C'].apply(lambda x: {'min': x.min(), 'max': x.max()})
    tm.assert_series_equal(result, expected)