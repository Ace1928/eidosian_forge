import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_multiindex_get_index_resolvers(self):
    df = DataFrame(np.ones((10, 3)), index=MultiIndex.from_arrays([range(10) for _ in range(2)], names=['spam', 'eggs']))
    resolvers = df._get_index_resolvers()

    def to_series(mi, level):
        level_values = mi.get_level_values(level)
        s = level_values.to_series()
        s.index = mi
        return s
    col_series = df.columns.to_series()
    expected = {'index': df.index, 'columns': col_series, 'spam': to_series(df.index, 'spam'), 'eggs': to_series(df.index, 'eggs'), 'clevel_0': col_series}
    for k, v in resolvers.items():
        if isinstance(v, Index):
            assert v.is_(expected[k])
        elif isinstance(v, Series):
            tm.assert_series_equal(v, expected[k])
        else:
            raise AssertionError('object must be a Series or Index')