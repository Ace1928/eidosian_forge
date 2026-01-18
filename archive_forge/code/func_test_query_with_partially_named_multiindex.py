import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_with_partially_named_multiindex(self, parser, engine):
    skip_if_no_pandas_parser(parser)
    a = np.random.default_rng(2).choice(['red', 'green'], size=10)
    b = np.arange(10)
    index = MultiIndex.from_arrays([a, b])
    index.names = [None, 'rating']
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
    res = df.query('rating == 1', parser=parser, engine=engine)
    ind = Series(df.index.get_level_values('rating').values, index=index, name='rating')
    exp = df[ind == 1]
    tm.assert_frame_equal(res, exp)
    res = df.query('rating != 1', parser=parser, engine=engine)
    ind = Series(df.index.get_level_values('rating').values, index=index, name='rating')
    exp = df[ind != 1]
    tm.assert_frame_equal(res, exp)
    res = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
    ind = Series(df.index.get_level_values(0).values, index=index)
    exp = df[ind == 'red']
    tm.assert_frame_equal(res, exp)
    res = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
    ind = Series(df.index.get_level_values(0).values, index=index)
    exp = df[ind != 'red']
    tm.assert_frame_equal(res, exp)