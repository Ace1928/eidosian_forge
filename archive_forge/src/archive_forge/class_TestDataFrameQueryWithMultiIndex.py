import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestDataFrameQueryWithMultiIndex:

    def test_query_with_named_multiindex(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(['red', 'green'], size=10)
        b = np.random.default_rng(2).choice(['eggs', 'ham'], size=10)
        index = MultiIndex.from_arrays([a, b], names=['color', 'food'])
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind = Series(df.index.get_level_values('color').values, index=index, name='color')
        res1 = df.query('color == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == color', parser=parser, engine=engine)
        exp = df[ind == 'red']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('color != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != color', parser=parser, engine=engine)
        exp = df[ind != 'red']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('color == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == color', parser=parser, engine=engine)
        exp = df[ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('color != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != color', parser=parser, engine=engine)
        exp = df[~ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["red"] in color', parser=parser, engine=engine)
        res2 = df.query('"red" in color', parser=parser, engine=engine)
        exp = df[ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["red"] not in color', parser=parser, engine=engine)
        res2 = df.query('"red" not in color', parser=parser, engine=engine)
        exp = df[~ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_unnamed_multiindex(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(['red', 'green'], size=10)
        b = np.random.default_rng(2).choice(['eggs', 'ham'], size=10)
        index = MultiIndex.from_arrays([a, b])
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind = Series(df.index.get_level_values(0).values, index=index)
        res1 = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == ilevel_0', parser=parser, engine=engine)
        exp = df[ind == 'red']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != ilevel_0', parser=parser, engine=engine)
        exp = df[ind != 'red']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_0 == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_0 != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["red"] in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" in ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["red"] not in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" not in ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(['red'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        ind = Series(df.index.get_level_values(1).values, index=index)
        res1 = df.query('ilevel_1 == "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" == ilevel_1', parser=parser, engine=engine)
        exp = df[ind == 'eggs']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_1 != "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" != ilevel_1', parser=parser, engine=engine)
        exp = df[ind != 'eggs']
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_1 == ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] == ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(['eggs'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('ilevel_1 != ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] != ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(['eggs'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["eggs"] in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" in ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(['eggs'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
        res1 = df.query('["eggs"] not in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" not in ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(['eggs'])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

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