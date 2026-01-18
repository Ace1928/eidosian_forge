from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('operation', ['agg', 'apply'])
@pytest.mark.parametrize('observed', [False, None])
def test_seriesgroupby_observed_false_or_none(df_cat, observed, operation):
    index, _ = MultiIndex.from_product([CategoricalIndex(['bar', 'foo'], ordered=False), CategoricalIndex(['one', 'three', 'two'], ordered=False)], names=['A', 'B']).sortlevel()
    expected = Series(data=[2, 4, np.nan, 1, np.nan, 3], index=index, name='C')
    if operation == 'agg':
        msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = expected.fillna(0, downcast='infer')
    grouped = df_cat.groupby(['A', 'B'], observed=observed)['C']
    msg = 'using SeriesGroupBy.sum' if operation == 'agg' else 'using np.sum'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)