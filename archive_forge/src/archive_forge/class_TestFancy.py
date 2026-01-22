import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
class TestFancy:
    """pure get/set item & fancy indexing"""

    def test_setitem_ndarray_1d(self):
        df = DataFrame(index=Index(np.arange(1, 11), dtype=np.int64))
        df['foo'] = np.zeros(10, dtype=np.float64)
        df['bar'] = np.zeros(10, dtype=complex)
        msg = 'Must have equal len keys and value when setting with an iterable'
        with pytest.raises(ValueError, match=msg):
            df.loc[df.index[2:5], 'bar'] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])
        df.loc[df.index[2:6], 'bar'] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])
        result = df.loc[df.index[2:6], 'bar']
        expected = Series([2.33j, 1.23 + 0.1j, 2.2, 1.0], index=[3, 4, 5, 6], name='bar')
        tm.assert_series_equal(result, expected)

    def test_setitem_ndarray_1d_2(self):
        df = DataFrame(index=Index(np.arange(1, 11)))
        df['foo'] = np.zeros(10, dtype=np.float64)
        df['bar'] = np.zeros(10, dtype=complex)
        msg = 'Must have equal len keys and value when setting with an iterable'
        with pytest.raises(ValueError, match=msg):
            df[2:5] = np.arange(1, 4) * 1j

    @pytest.mark.filterwarnings('ignore:Series.__getitem__ treating keys as positions is deprecated:FutureWarning')
    def test_getitem_ndarray_3d(self, index, frame_or_series, indexer_sli, using_array_manager):
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))
        msgs = []
        if frame_or_series is Series and indexer_sli in [tm.setitem, tm.iloc]:
            msgs.append('Wrong number of dimensions. values.ndim > ndim \\[3 > 1\\]')
            if using_array_manager:
                msgs.append('Passed array should be 1-dimensional')
        if frame_or_series is Series or indexer_sli is tm.iloc:
            msgs.append('Buffer has wrong number of dimensions \\(expected 1, got 3\\)')
            if using_array_manager:
                msgs.append('indexer should be 1-dimensional')
        if indexer_sli is tm.loc or (frame_or_series is Series and indexer_sli is tm.setitem):
            msgs.append('Cannot index with multidimensional key')
        if frame_or_series is DataFrame and indexer_sli is tm.setitem:
            msgs.append('Index data must be 1-dimensional')
        if isinstance(index, pd.IntervalIndex) and indexer_sli is tm.iloc:
            msgs.append('Index data must be 1-dimensional')
        if isinstance(index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex)):
            msgs.append('Data must be 1-dimensional')
        if len(index) == 0 or isinstance(index, pd.MultiIndex):
            msgs.append('positional indexers are out-of-bounds')
        if type(index) is Index and (not isinstance(index._values, np.ndarray)):
            msgs.append('values must be a 1D array')
            msgs.append('only handle 1-dimensional arrays')
        msg = '|'.join(msgs)
        potential_errors = (IndexError, ValueError, NotImplementedError)
        with pytest.raises(potential_errors, match=msg):
            idxr[nd3]

    @pytest.mark.filterwarnings('ignore:Series.__setitem__ treating keys as positions is deprecated:FutureWarning')
    def test_setitem_ndarray_3d(self, index, frame_or_series, indexer_sli):
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))
        if indexer_sli is tm.iloc:
            err = ValueError
            msg = f'Cannot set values with ndim > {obj.ndim}'
        else:
            err = ValueError
            msg = '|'.join(['Buffer has wrong number of dimensions \\(expected 1, got 3\\)', 'Cannot set values with ndim > 1', 'Index data must be 1-dimensional', 'Data must be 1-dimensional', 'Array conditional must be same shape as self'])
        with pytest.raises(err, match=msg):
            idxr[nd3] = 0

    def test_getitem_ndarray_0d(self):
        key = np.array(0)
        df = DataFrame([[1, 2], [3, 4]])
        result = df[key]
        expected = Series([1, 3], name=0)
        tm.assert_series_equal(result, expected)
        ser = Series([1, 2])
        result = ser[key]
        assert result == 1

    def test_inf_upcast(self):
        df = DataFrame(columns=[0])
        df.loc[1] = 1
        df.loc[2] = 2
        df.loc[np.inf] = 3
        assert df.loc[np.inf, 0] == 3
        result = df.index
        expected = Index([1, 2, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    def test_setitem_dtype_upcast(self):
        df = DataFrame([{'a': 1}, {'a': 3, 'b': 2}])
        df['c'] = np.nan
        assert df['c'].dtype == np.float64
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            df.loc[0, 'c'] = 'foo'
        expected = DataFrame({'a': [1, 3], 'b': [np.nan, 2], 'c': Series(['foo', np.nan], dtype=object)})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('val', [3.14, 'wxyz'])
    def test_setitem_dtype_upcast2(self, val):
        df = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=list('ab'), columns=['foo', 'bar', 'baz'])
        left = df.copy()
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            left.loc['a', 'bar'] = val
        right = DataFrame([[0, val, 2], [3, 4, 5]], index=list('ab'), columns=['foo', 'bar', 'baz'])
        tm.assert_frame_equal(left, right)
        assert is_integer_dtype(left['foo'])
        assert is_integer_dtype(left['baz'])

    def test_setitem_dtype_upcast3(self):
        left = DataFrame(np.arange(6, dtype='int64').reshape(2, 3) / 10.0, index=list('ab'), columns=['foo', 'bar', 'baz'])
        with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
            left.loc['a', 'bar'] = 'wxyz'
        right = DataFrame([[0, 'wxyz', 0.2], [0.3, 0.4, 0.5]], index=list('ab'), columns=['foo', 'bar', 'baz'])
        tm.assert_frame_equal(left, right)
        assert is_float_dtype(left['foo'])
        assert is_float_dtype(left['baz'])

    def test_dups_fancy_indexing(self):
        df = DataFrame(np.eye(3), columns=['a', 'a', 'b'])
        result = df[['b', 'a']].columns
        expected = Index(['b', 'a', 'a'])
        tm.assert_index_equal(result, expected)

    def test_dups_fancy_indexing_across_dtypes(self):
        df = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']], columns=list('aaaaaaa'))
        result = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']])
        result.columns = list('aaaaaaa')
        df.iloc[:, 4]
        result.iloc[:, 4]
        tm.assert_frame_equal(df, result)

    def test_dups_fancy_indexing_not_in_order(self):
        df = DataFrame({'test': [5, 7, 9, 11], 'test1': [4.0, 5, 6, 7], 'other': list('abcd')}, index=['A', 'A', 'B', 'C'])
        rows = ['C', 'B']
        expected = DataFrame({'test': [11, 9], 'test1': [7.0, 6], 'other': ['d', 'c']}, index=rows)
        result = df.loc[rows]
        tm.assert_frame_equal(result, expected)
        result = df.loc[Index(rows)]
        tm.assert_frame_equal(result, expected)
        rows = ['C', 'B', 'E']
        with pytest.raises(KeyError, match='not in index'):
            df.loc[rows]
        rows = ['F', 'G', 'H', 'C', 'B', 'E']
        with pytest.raises(KeyError, match='not in index'):
            df.loc[rows]

    def test_dups_fancy_indexing_only_missing_label(self, using_infer_string):
        dfnu = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=list('AABCD'))
        if using_infer_string:
            with pytest.raises(KeyError, match=re.escape('"None of [Index([\'E\'], dtype=\'string\')] are in the [index]"')):
                dfnu.loc[['E']]
        else:
            with pytest.raises(KeyError, match=re.escape('"None of [Index([\'E\'], dtype=\'object\')] are in the [index]"')):
                dfnu.loc[['E']]

    @pytest.mark.parametrize('vals', [[0, 1, 2], list('abc')])
    def test_dups_fancy_indexing_missing_label(self, vals):
        df = DataFrame({'A': vals})
        with pytest.raises(KeyError, match='not in index'):
            df.loc[[0, 8, 0]]

    def test_dups_fancy_indexing_non_unique(self):
        df = DataFrame({'test': [5, 7, 9, 11]}, index=['A', 'A', 'B', 'C'])
        with pytest.raises(KeyError, match='not in index'):
            df.loc[['A', 'A', 'E']]

    def test_dups_fancy_indexing2(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), columns=['A', 'B', 'B', 'B', 'A'])
        with pytest.raises(KeyError, match='not in index'):
            df.loc[:, ['A', 'B', 'C']]

    def test_dups_fancy_indexing3(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((9, 2)), index=[1, 1, 1, 2, 2, 2, 3, 3, 3], columns=['a', 'b'])
        expected = df.iloc[0:6]
        result = df.loc[[1, 2]]
        tm.assert_frame_equal(result, expected)
        expected = df
        result = df.loc[:, ['a', 'b']]
        tm.assert_frame_equal(result, expected)
        expected = df.iloc[0:6, :]
        result = df.loc[[1, 2], ['a', 'b']]
        tm.assert_frame_equal(result, expected)

    def test_duplicate_int_indexing(self, indexer_sl):
        ser = Series(range(3), index=[1, 1, 3])
        expected = Series(range(2), index=[1, 1])
        result = indexer_sl(ser)[[1]]
        tm.assert_series_equal(result, expected)

    def test_indexing_mixed_frame_bug(self):
        df = DataFrame({'a': {1: 'aaa', 2: 'bbb', 3: 'ccc'}, 'b': {1: 111, 2: 222, 3: 333}})
        df['test'] = df['a'].apply(lambda x: '_' if x == 'aaa' else x)
        idx = df['test'] == '_'
        temp = df.loc[idx, 'a'].apply(lambda x: '-----' if x == 'aaa' else x)
        df.loc[idx, 'test'] = temp
        assert df.iloc[0, 2] == '-----'

    def test_multitype_list_index_access(self):
        df = DataFrame(np.random.default_rng(2).random((10, 5)), columns=['a'] + [20, 21, 22, 23])
        with pytest.raises(KeyError, match=re.escape("'[26, -8] not in index'")):
            df[[22, 26, -8]]
        assert df[21].shape[0] == df.shape[0]

    def test_set_index_nan(self):
        df = DataFrame({'PRuid': {17: 'nonQC', 18: 'nonQC', 19: 'nonQC', 20: '10', 21: '11', 22: '12', 23: '13', 24: '24', 25: '35', 26: '46', 27: '47', 28: '48', 29: '59', 30: '10'}, 'QC': {17: 0.0, 18: 0.0, 19: 0.0, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: 1.0, 25: np.nan, 26: np.nan, 27: np.nan, 28: np.nan, 29: np.nan, 30: np.nan}, 'data': {17: 7.95449, 18: 8.014261, 19: 7.859152000000001, 20: 0.8614035, 21: 0.8785311, 22: 0.8427041999999999, 23: 0.785877, 24: 0.7306246, 25: 0.8166856, 26: 0.8192708000000001, 27: 0.8070501, 28: 0.8144024000000001, 29: 0.8014085, 30: 0.8130774000000001}, 'year': {17: 2006, 18: 2007, 19: 2008, 20: 1985, 21: 1985, 22: 1985, 23: 1985, 24: 1985, 25: 1985, 26: 1985, 27: 1985, 28: 1985, 29: 1985, 30: 1986}}).reset_index()
        result = df.set_index(['year', 'PRuid', 'QC']).reset_index().reindex(columns=df.columns)
        tm.assert_frame_equal(result, df)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't multiply arrow strings")
    def test_multi_assign(self):
        df = DataFrame({'FC': ['a', 'b', 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': list(range(6)), 'col2': list(range(6, 12))}).astype({'col2': 'float64'})
        df.iloc[1, 0] = np.nan
        df2 = df.copy()
        mask = ~df2.FC.isna()
        cols = ['col1', 'col2']
        dft = df2 * 2
        dft.iloc[3, 3] = np.nan
        expected = DataFrame({'FC': ['a', np.nan, 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': Series([0, 1, 4, 6, 8, 10]), 'col2': [12, 7, 16, np.nan, 20, 22]})
        df2.loc[mask, cols] = dft.loc[mask, cols]
        tm.assert_frame_equal(df2, expected)
        expected = DataFrame({'FC': ['a', np.nan, 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': [0, 1, 4, 6, 8, 10], 'col2': [12, 7, 16, np.nan, 20, 22]})
        df2 = df.copy()
        df2.loc[mask, cols] = dft.loc[mask, cols].values
        tm.assert_frame_equal(df2, expected)

    def test_multi_assign_broadcasting_rhs(self):
        df = DataFrame({'A': [1, 2, 0, 0, 0], 'B': [0, 0, 0, 10, 11], 'C': [0, 0, 0, 10, 11], 'D': [3, 4, 5, 6, 7]})
        expected = df.copy()
        mask = expected['A'] == 0
        for col in ['A', 'B']:
            expected.loc[mask, col] = df['D']
        df.loc[df['A'] == 0, ['A', 'B']] = df['D'].copy()
        tm.assert_frame_equal(df, expected)

    def test_setitem_list(self):
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = [1, 2, 3]
        df.iloc[1, 0] = [1, 2]
        result = DataFrame(index=[0, 1], columns=[0])
        result.iloc[1, 0] = [1, 2]
        tm.assert_frame_equal(result, df)

    def test_string_slice(self):
        df = DataFrame([1], Index([pd.Timestamp('2011-01-01')], dtype=object))
        assert df.index._is_all_dates
        with pytest.raises(KeyError, match="'2011'"):
            df['2011']
        with pytest.raises(KeyError, match="'2011'"):
            df.loc['2011', 0]

    def test_string_slice_empty(self):
        df = DataFrame()
        assert not df.index._is_all_dates
        with pytest.raises(KeyError, match="'2011'"):
            df['2011']
        with pytest.raises(KeyError, match='^0$'):
            df.loc['2011', 0]

    def test_astype_assignment(self, using_infer_string):
        df_orig = DataFrame([['1', '2', '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        df = df_orig.copy()
        df.iloc[:, 0:2] = df.iloc[:, 0:2].astype(np.int64)
        expected = DataFrame([[1, 2, '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        if not using_infer_string:
            expected['A'] = expected['A'].astype(object)
            expected['B'] = expected['B'].astype(object)
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.loc[:, 'A'] = df.loc[:, 'A'].astype(np.int64)
        expected = DataFrame([[1, '2', '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        if not using_infer_string:
            expected['A'] = expected['A'].astype(object)
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.loc[:, ['B', 'C']] = df.loc[:, ['B', 'C']].astype(np.int64)
        expected = DataFrame([['1', 2, 3, '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        if not using_infer_string:
            expected['B'] = expected['B'].astype(object)
            expected['C'] = expected['C'].astype(object)
        tm.assert_frame_equal(df, expected)

    def test_astype_assignment_full_replacements(self):
        df = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        df.iloc[:, 0] = df['A'].astype(np.int64)
        expected = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        df.loc[:, 'A'] = df['A'].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', [tm.getitem, tm.loc])
    def test_index_type_coercion(self, indexer):
        for s in [Series(range(5)), Series(range(5), index=range(1, 6))]:
            assert is_integer_dtype(s.index)
            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert is_float_dtype(s2.index)
            assert indexer(s2)[0.1] == 0
            s2 = s.copy()
            indexer(s2)[0.0] = 0
            exp = s.index
            if 0 not in s:
                exp = Index(s.index.tolist() + [0])
            tm.assert_index_equal(s2.index, exp)
            s2 = s.copy()
            indexer(s2)['0'] = 0
            assert is_object_dtype(s2.index)
        for s in [Series(range(5), index=np.arange(5.0))]:
            assert is_float_dtype(s.index)
            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert is_float_dtype(s2.index)
            assert indexer(s2)[0.1] == 0
            s2 = s.copy()
            indexer(s2)[0.0] = 0
            tm.assert_index_equal(s2.index, s.index)
            s2 = s.copy()
            indexer(s2)['0'] = 0
            assert is_object_dtype(s2.index)