from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TestLocWithMultiIndex:

    @pytest.mark.parametrize('keys, expected', [(['b', 'a'], [['b', 'b', 'a', 'a'], [1, 2, 1, 2]]), (['a', 'b'], [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]), ((['a', 'b'], [1, 2]), [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]), ((['a', 'b'], [2, 1]), [['a', 'a', 'b', 'b'], [2, 1, 2, 1]]), ((['b', 'a'], [2, 1]), [['b', 'b', 'a', 'a'], [2, 1, 2, 1]]), ((['b', 'a'], [1, 2]), [['b', 'b', 'a', 'a'], [1, 2, 1, 2]]), ((['c', 'a'], [2, 1]), [['c', 'a', 'a'], [1, 2, 1]])])
    @pytest.mark.parametrize('dim', ['index', 'columns'])
    def test_loc_getitem_multilevel_index_order(self, dim, keys, expected):
        kwargs = {dim: [['c', 'a', 'a', 'b', 'b'], [1, 1, 2, 1, 2]]}
        df = DataFrame(np.arange(25).reshape(5, 5), **kwargs)
        exp_index = MultiIndex.from_arrays(expected)
        if dim == 'index':
            res = df.loc[keys, :]
            tm.assert_index_equal(res.index, exp_index)
        elif dim == 'columns':
            res = df.loc[:, keys]
            tm.assert_index_equal(res.columns, exp_index)

    def test_loc_preserve_names(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data
        result = ymd.loc[2000]
        result2 = ymd['A'].loc[2000]
        assert result.index.names == ymd.index.names[1:]
        assert result2.index.names == ymd.index.names[1:]
        result = ymd.loc[2000, 2]
        result2 = ymd['A'].loc[2000, 2]
        assert result.index.name == ymd.index.names[2]
        assert result2.index.name == ymd.index.names[2]

    def test_loc_getitem_multiindex_nonunique_len_zero(self):
        mi = MultiIndex.from_product([[0], [1, 1]])
        ser = Series(0, index=mi)
        res = ser.loc[[]]
        expected = ser[:0]
        tm.assert_series_equal(res, expected)
        res2 = ser.loc[ser.iloc[0:0]]
        tm.assert_series_equal(res2, expected)

    def test_loc_getitem_access_none_value_in_multiindex(self):
        ser = Series([None], MultiIndex.from_arrays([['Level1'], ['Level2']]))
        result = ser.loc['Level1', 'Level2']
        assert result is None
        midx = MultiIndex.from_product([['Level1'], ['Level2_a', 'Level2_b']])
        ser = Series([None] * len(midx), dtype=object, index=midx)
        result = ser.loc['Level1', 'Level2_a']
        assert result is None
        ser = Series([1] * len(midx), dtype=object, index=midx)
        result = ser.loc['Level1', 'Level2_a']
        assert result == 1

    def test_loc_setitem_multiindex_slice(self):
        index = MultiIndex.from_tuples(zip(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']), names=['first', 'second'])
        result = Series([1, 1, 1, 1, 1, 1, 1, 1], index=index)
        result.loc[('baz', 'one'):('foo', 'two')] = 100
        expected = Series([1, 1, 100, 100, 100, 100, 1, 1], index=index)
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_slice_datetime_objs_with_datetimeindex(self):
        times = date_range('2000-01-01', freq='10min', periods=100000)
        ser = Series(range(100000), times)
        result = ser.loc[datetime(1900, 1, 1):datetime(2100, 1, 1)]
        tm.assert_series_equal(result, ser)

    def test_loc_getitem_datetime_string_with_datetimeindex(self):
        df = DataFrame({'a': range(10), 'b': range(10)}, index=date_range('2010-01-01', '2010-01-10'))
        result = df.loc[['2010-01-01', '2010-01-05'], ['a', 'b']]
        expected = DataFrame({'a': [0, 4], 'b': [0, 4]}, index=DatetimeIndex(['2010-01-01', '2010-01-05']))
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sorted_index_level_with_duplicates(self):
        mi = MultiIndex.from_tuples([('foo', 'bar'), ('foo', 'bar'), ('bah', 'bam'), ('bah', 'bam'), ('foo', 'bar'), ('bah', 'bam')], names=['A', 'B'])
        df = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3], [4.0, 4], [5.0, 5], [6.0, 6]], index=mi, columns=['C', 'D'])
        df = df.sort_index(level=0)
        expected = DataFrame([[1.0, 1], [2.0, 2], [5.0, 5]], columns=['C', 'D'], index=mi.take([0, 1, 4]))
        result = df.loc['foo', 'bar']
        tm.assert_frame_equal(result, expected)

    def test_additional_element_to_categorical_series_loc(self):
        result = Series(['a', 'b', 'c'], dtype='category')
        result.loc[3] = 0
        expected = Series(['a', 'b', 'c', 0], dtype='object')
        tm.assert_series_equal(result, expected)

    def test_additional_categorical_element_loc(self):
        result = Series(['a', 'b', 'c'], dtype='category')
        result.loc[3] = 'a'
        expected = Series(['a', 'b', 'c', 'a'], dtype='category')
        tm.assert_series_equal(result, expected)

    def test_loc_set_nan_in_categorical_series(self, any_numeric_ea_dtype):
        srs = Series([1, 2, 3], dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
        srs.loc[3] = np.nan
        expected = Series([1, 2, 3, np.nan], dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
        tm.assert_series_equal(srs, expected)
        srs.loc[1] = np.nan
        expected = Series([1, np.nan, 3, np.nan], dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
        tm.assert_series_equal(srs, expected)

    @pytest.mark.parametrize('na', (np.nan, pd.NA, None, pd.NaT))
    def test_loc_consistency_series_enlarge_set_into(self, na):
        srs_enlarge = Series(['a', 'b', 'c'], dtype='category')
        srs_enlarge.loc[3] = na
        srs_setinto = Series(['a', 'b', 'c', 'a'], dtype='category')
        srs_setinto.loc[3] = na
        tm.assert_series_equal(srs_enlarge, srs_setinto)
        expected = Series(['a', 'b', 'c', na], dtype='category')
        tm.assert_series_equal(srs_enlarge, expected)

    def test_loc_getitem_preserves_index_level_category_dtype(self):
        df = DataFrame(data=np.arange(2, 22, 2), index=MultiIndex(levels=[CategoricalIndex(['a', 'b']), range(10)], codes=[[0] * 5 + [1] * 5, range(10)], names=['Index1', 'Index2']))
        expected = CategoricalIndex(['a', 'b'], categories=['a', 'b'], ordered=False, name='Index1', dtype='category')
        result = df.index.levels[0]
        tm.assert_index_equal(result, expected)
        result = df.loc[['a']].index.levels[0]
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('lt_value', [30, 10])
    def test_loc_multiindex_levels_contain_values_not_in_index_anymore(self, lt_value):
        df = DataFrame({'a': [12, 23, 34, 45]}, index=[list('aabb'), [0, 1, 2, 3]])
        with pytest.raises(KeyError, match="\\['b'\\] not in index"):
            df.loc[df['a'] < lt_value, :].loc[['b'], :]

    def test_loc_multiindex_null_slice_na_level(self):
        lev1 = np.array([np.nan, np.nan])
        lev2 = ['bar', 'baz']
        mi = MultiIndex.from_arrays([lev1, lev2])
        ser = Series([0, 1], index=mi)
        result = ser.loc[:, 'bar']
        expected = Series([0], index=[np.nan])
        tm.assert_series_equal(result, expected)

    def test_loc_drops_level(self):
        mi = MultiIndex.from_product([list('ab'), list('xy'), [1, 2]], names=['ab', 'xy', 'num'])
        ser = Series(range(8), index=mi)
        loc_result = ser.loc['a', :, :]
        expected = ser.index.droplevel(0)[:4]
        tm.assert_index_equal(loc_result.index, expected)