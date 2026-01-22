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
class TestLocSeries:

    @pytest.mark.parametrize('val,expected', [(2 ** 63 - 1, 3), (2 ** 63, 4)])
    def test_loc_uint64(self, val, expected):
        ser = Series({2 ** 63 - 1: 3, 2 ** 63: 4})
        assert ser.loc[val] == expected

    def test_loc_getitem(self, string_series, datetime_series):
        inds = string_series.index[[3, 4, 7]]
        tm.assert_series_equal(string_series.loc[inds], string_series.reindex(inds))
        tm.assert_series_equal(string_series.iloc[5::2], string_series[5::2])
        d1, d2 = datetime_series.index[[5, 15]]
        result = datetime_series.loc[d1:d2]
        expected = datetime_series.truncate(d1, d2)
        tm.assert_series_equal(result, expected)
        mask = string_series > string_series.median()
        tm.assert_series_equal(string_series.loc[mask], string_series[mask])
        assert datetime_series.loc[d1] == datetime_series[d1]
        assert datetime_series.loc[d2] == datetime_series[d2]

    def test_loc_getitem_not_monotonic(self, datetime_series):
        d1, d2 = datetime_series.index[[5, 15]]
        ts2 = datetime_series[::2].iloc[[1, 2, 0]]
        msg = "Timestamp\\('2000-01-10 00:00:00'\\)"
        with pytest.raises(KeyError, match=msg):
            ts2.loc[d1:d2]
        with pytest.raises(KeyError, match=msg):
            ts2.loc[d1:d2] = 0

    def test_loc_getitem_setitem_integer_slice_keyerrors(self):
        ser = Series(np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2)))
        cp = ser.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).all()
        cp = ser.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()
        result = ser.iloc[2:6]
        result2 = ser.loc[3:11]
        expected = ser.reindex([4, 6, 8, 10])
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)
        s2 = ser.iloc[list(range(5)) + list(range(9, 4, -1))]
        with pytest.raises(KeyError, match='^3$'):
            s2.loc[3:11]
        with pytest.raises(KeyError, match='^3$'):
            s2.loc[3:11] = 0

    def test_loc_getitem_iterator(self, string_series):
        idx = iter(string_series.index[:10])
        result = string_series.loc[idx]
        tm.assert_series_equal(result, string_series[:10])

    def test_loc_setitem_boolean(self, string_series):
        mask = string_series > string_series.median()
        result = string_series.copy()
        result.loc[mask] = 0
        expected = string_series
        expected[mask] = 0
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_corner(self, string_series):
        inds = list(string_series.index[[5, 8, 12]])
        string_series.loc[inds] = 5
        msg = "\\['foo'\\] not in index"
        with pytest.raises(KeyError, match=msg):
            string_series.loc[inds + ['foo']] = 5

    def test_basic_setitem_with_labels(self, datetime_series):
        indices = datetime_series.index[[5, 10, 15]]
        cp = datetime_series.copy()
        exp = datetime_series.copy()
        cp[indices] = 0
        exp.loc[indices] = 0
        tm.assert_series_equal(cp, exp)
        cp = datetime_series.copy()
        exp = datetime_series.copy()
        cp[indices[0]:indices[2]] = 0
        exp.loc[indices[0]:indices[2]] = 0
        tm.assert_series_equal(cp, exp)

    def test_loc_setitem_listlike_of_ints(self):
        ser = Series(np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2)))
        inds = [0, 4, 6]
        arr_inds = np.array([0, 4, 6])
        cp = ser.copy()
        exp = ser.copy()
        ser[inds] = 0
        ser.loc[inds] = 0
        tm.assert_series_equal(cp, exp)
        cp = ser.copy()
        exp = ser.copy()
        ser[arr_inds] = 0
        ser.loc[arr_inds] = 0
        tm.assert_series_equal(cp, exp)
        inds_notfound = [0, 4, 5, 6]
        arr_inds_notfound = np.array([0, 4, 5, 6])
        msg = '\\[5\\] not in index'
        with pytest.raises(KeyError, match=msg):
            ser[inds_notfound] = 0
        with pytest.raises(Exception, match=msg):
            ser[arr_inds_notfound] = 0

    def test_loc_setitem_dt64tz_values(self):
        ser = Series(date_range('2011-01-01', periods=3, tz='US/Eastern'), index=['a', 'b', 'c'])
        s2 = ser.copy()
        expected = Timestamp('2011-01-03', tz='US/Eastern')
        s2.loc['a'] = expected
        result = s2.loc['a']
        assert result == expected
        s2 = ser.copy()
        s2.iloc[0] = expected
        result = s2.iloc[0]
        assert result == expected
        s2 = ser.copy()
        s2['a'] = expected
        result = s2['a']
        assert result == expected

    @pytest.mark.parametrize('array_fn', [np.array, pd.array, list, tuple])
    @pytest.mark.parametrize('size', [0, 4, 5, 6])
    def test_loc_iloc_setitem_with_listlike(self, size, array_fn):
        arr = array_fn([0] * size)
        expected = Series([arr, 0, 0, 0, 0], index=list('abcde'), dtype=object)
        ser = Series(0, index=list('abcde'), dtype=object)
        ser.loc['a'] = arr
        tm.assert_series_equal(ser, expected)
        ser = Series(0, index=list('abcde'), dtype=object)
        ser.iloc[0] = arr
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize('indexer', [IndexSlice['A', :], ('A', slice(None))])
    def test_loc_series_getitem_too_many_dimensions(self, indexer):
        ser = Series(index=MultiIndex.from_tuples([('A', '0'), ('A', '1'), ('B', '0')]), data=[21, 22, 23])
        msg = 'Too many indexers'
        with pytest.raises(IndexingError, match=msg):
            ser.loc[indexer, :]
        with pytest.raises(IndexingError, match=msg):
            ser.loc[indexer, :] = 1

    def test_loc_setitem(self, string_series):
        inds = string_series.index[[3, 4, 7]]
        result = string_series.copy()
        result.loc[inds] = 5
        expected = string_series.copy()
        expected.iloc[[3, 4, 7]] = 5
        tm.assert_series_equal(result, expected)
        result.iloc[5:10] = 10
        expected[5:10] = 10
        tm.assert_series_equal(result, expected)
        d1, d2 = string_series.index[[5, 15]]
        result.loc[d1:d2] = 6
        expected[5:16] = 6
        tm.assert_series_equal(result, expected)
        string_series.loc[d1] = 4
        string_series.loc[d2] = 6
        assert string_series[d1] == 4
        assert string_series[d2] == 6

    @pytest.mark.parametrize('dtype', ['object', 'string'])
    def test_loc_assign_dict_to_row(self, dtype):
        df = DataFrame({'A': ['abc', 'def'], 'B': ['ghi', 'jkl']}, dtype=dtype)
        df.loc[0, :] = {'A': 'newA', 'B': 'newB'}
        expected = DataFrame({'A': ['newA', 'def'], 'B': ['newB', 'jkl']}, dtype=dtype)
        tm.assert_frame_equal(df, expected)

    @td.skip_array_manager_invalid_test
    def test_loc_setitem_dict_timedelta_multiple_set(self):
        result = DataFrame(columns=['time', 'value'])
        result.loc[1] = {'time': Timedelta(6, unit='s'), 'value': 'foo'}
        result.loc[1] = {'time': Timedelta(6, unit='s'), 'value': 'foo'}
        expected = DataFrame([[Timedelta(6, unit='s'), 'foo']], columns=['time', 'value'], index=[1])
        tm.assert_frame_equal(result, expected)

    def test_loc_set_multiple_items_in_multiple_new_columns(self):
        df = DataFrame(index=[1, 2], columns=['a'])
        df.loc[1, ['b', 'c']] = [6, 7]
        expected = DataFrame({'a': Series([np.nan, np.nan], dtype='object'), 'b': [6, np.nan], 'c': [7, np.nan]}, index=[1, 2])
        tm.assert_frame_equal(df, expected)

    def test_getitem_loc_str_periodindex(self):
        msg = 'Period with BDay freq is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            index = pd.period_range(start='2000', periods=20, freq='B')
            series = Series(range(20), index=index)
            assert series.loc['2000-01-14'] == 9

    def test_loc_nonunique_masked_index(self):
        ids = list(range(11))
        index = Index(ids * 1000, dtype='Int64')
        df = DataFrame({'val': np.arange(len(index), dtype=np.intp)}, index=index)
        result = df.loc[ids]
        expected = DataFrame({'val': index.argsort(kind='stable').astype(np.intp)}, index=Index(np.array(ids).repeat(1000), dtype='Int64'))
        tm.assert_frame_equal(result, expected)