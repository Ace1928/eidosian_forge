from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
@pytest.mark.parametrize('values', [['bar', 'baz'], np.array(['bar', 'baz']), Series(['bar', 'baz']), Index(['bar', 'baz'])])
@pytest.mark.parametrize('method', [True, False])
def test_pivot_with_list_like_values_nans(self, values, method):
    df = DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    if method:
        result = df.pivot(index='zoo', columns='foo', values=values)
    else:
        result = pd.pivot(df, index='zoo', columns='foo', values=values)
    data = [[np.nan, 'A', np.nan, 4], [np.nan, 'C', np.nan, 6], [np.nan, 'B', np.nan, 5], ['A', np.nan, 1, np.nan], ['B', np.nan, 2, np.nan], ['C', np.nan, 3, np.nan]]
    index = Index(data=['q', 't', 'w', 'x', 'y', 'z'], name='zoo')
    columns = MultiIndex(levels=[['bar', 'baz'], ['one', 'two']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]], names=[None, 'foo'])
    expected = DataFrame(data=data, index=index, columns=columns)
    expected['baz'] = expected['baz'].astype(object)
    tm.assert_frame_equal(result, expected)