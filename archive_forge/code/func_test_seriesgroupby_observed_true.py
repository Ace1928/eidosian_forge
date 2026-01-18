from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('operation', ['agg', 'apply'])
def test_seriesgroupby_observed_true(df_cat, operation):
    lev_a = Index(['bar', 'bar', 'foo', 'foo'], dtype=df_cat['A'].dtype, name='A')
    lev_b = Index(['one', 'three', 'one', 'two'], dtype=df_cat['B'].dtype, name='B')
    index = MultiIndex.from_arrays([lev_a, lev_b])
    expected = Series(data=[2, 4, 1, 3], index=index, name='C').sort_index()
    grouped = df_cat.groupby(['A', 'B'], observed=True)['C']
    msg = 'using np.sum' if operation == 'apply' else 'using SeriesGroupBy.sum'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)