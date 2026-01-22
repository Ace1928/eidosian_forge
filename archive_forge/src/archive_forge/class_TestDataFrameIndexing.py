from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDataFrameIndexing:

    def test_getitem(self, float_frame):
        sl = float_frame[:20]
        assert len(sl.index) == 20
        for _, series in sl.items():
            assert len(series.index) == 20
            tm.assert_index_equal(series.index, sl.index)
        for key, _ in float_frame._series.items():
            assert float_frame[key] is not None
        assert 'random' not in float_frame
        with pytest.raises(KeyError, match='random'):
            float_frame['random']

    def test_getitem_numeric_should_not_fallback_to_positional(self, any_numeric_dtype):
        dtype = any_numeric_dtype
        idx = Index([1, 0, 1], dtype=dtype)
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=idx)
        result = df[1]
        expected = DataFrame([[1, 3], [4, 6]], columns=Index([1, 1], dtype=dtype))
        tm.assert_frame_equal(result, expected, check_exact=True)

    def test_getitem2(self, float_frame):
        df = float_frame.copy()
        df['$10'] = np.random.default_rng(2).standard_normal(len(df))
        ad = np.random.default_rng(2).standard_normal(len(df))
        df['@awesome_domain'] = ad
        with pytest.raises(KeyError, match=re.escape('\'df["$10"]\'')):
            df.__getitem__('df["$10"]')
        res = df['@awesome_domain']
        tm.assert_numpy_array_equal(ad, res.values)

    def test_setitem_numeric_should_not_fallback_to_positional(self, any_numeric_dtype):
        dtype = any_numeric_dtype
        idx = Index([1, 0, 1], dtype=dtype)
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=idx)
        df[1] = 10
        expected = DataFrame([[10, 2, 10], [10, 5, 10]], columns=idx)
        tm.assert_frame_equal(df, expected, check_exact=True)

    def test_setitem_list(self, float_frame):
        float_frame['E'] = 'foo'
        data = float_frame[['A', 'B']]
        float_frame[['B', 'A']] = data
        tm.assert_series_equal(float_frame['B'], data['A'], check_names=False)
        tm.assert_series_equal(float_frame['A'], data['B'], check_names=False)
        msg = 'Columns must be same length as key'
        with pytest.raises(ValueError, match=msg):
            data[['A']] = float_frame[['A', 'B']]
        newcolumndata = range(len(data.index) - 1)
        msg = f'Length of values \\({len(newcolumndata)}\\) does not match length of index \\({len(data)}\\)'
        with pytest.raises(ValueError, match=msg):
            data['A'] = newcolumndata

    def test_setitem_list2(self):
        df = DataFrame(0, index=range(3), columns=['tt1', 'tt2'], dtype=int)
        df.loc[1, ['tt1', 'tt2']] = [1, 2]
        result = df.loc[df.index[1], ['tt1', 'tt2']]
        expected = Series([1, 2], df.columns, dtype=int, name=1)
        tm.assert_series_equal(result, expected)
        df['tt1'] = df['tt2'] = '0'
        df.loc[df.index[1], ['tt1', 'tt2']] = ['1', '2']
        result = df.loc[df.index[1], ['tt1', 'tt2']]
        expected = Series(['1', '2'], df.columns, name=1)
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean(self, mixed_float_frame, mixed_int_frame, datetime_frame):
        d = datetime_frame.index[10]
        indexer = datetime_frame.index > d
        indexer_obj = indexer.astype(object)
        subindex = datetime_frame.index[indexer]
        subframe = datetime_frame[indexer]
        tm.assert_index_equal(subindex, subframe.index)
        with pytest.raises(ValueError, match='Item wrong length'):
            datetime_frame[indexer[:-1]]
        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)
        with pytest.raises(ValueError, match='Boolean array expected'):
            datetime_frame[datetime_frame]
        indexer_obj = Series(indexer_obj, datetime_frame.index)
        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)
        with tm.assert_produces_warning(UserWarning):
            indexer_obj = indexer_obj.reindex(datetime_frame.index[::-1])
            subframe_obj = datetime_frame[indexer_obj]
            tm.assert_frame_equal(subframe_obj, subframe)
        for df in [datetime_frame, mixed_float_frame, mixed_int_frame]:
            data = df._get_numeric_data()
            bif = df[df > 0]
            bifw = DataFrame({c: np.where(data[c] > 0, data[c], np.nan) for c in data.columns}, index=data.index, columns=data.columns)
            for c in df.columns:
                if c not in bifw:
                    bifw[c] = df[c]
            bifw = bifw.reindex(columns=df.columns)
            tm.assert_frame_equal(bif, bifw, check_dtype=False)
            for c in df.columns:
                if bif[c].dtype != bifw[c].dtype:
                    assert bif[c].dtype == df[c].dtype

    def test_getitem_boolean_casting(self, datetime_frame):
        df = datetime_frame.copy()
        df['E'] = 1
        df['E'] = df['E'].astype('int32')
        df['E1'] = df['E'].copy()
        df['F'] = 1
        df['F'] = df['F'].astype('int64')
        df['F1'] = df['F'].copy()
        casted = df[df > 0]
        result = casted.dtypes
        expected = Series([np.dtype('float64')] * 4 + [np.dtype('int32')] * 2 + [np.dtype('int64')] * 2, index=['A', 'B', 'C', 'D', 'E', 'E1', 'F', 'F1'])
        tm.assert_series_equal(result, expected)
        df.loc[df.index[1:3], ['E1', 'F1']] = 0
        casted = df[df > 0]
        result = casted.dtypes
        expected = Series([np.dtype('float64')] * 4 + [np.dtype('int32')] + [np.dtype('float64')] + [np.dtype('int64')] + [np.dtype('float64')], index=['A', 'B', 'C', 'D', 'E', 'E1', 'F', 'F1'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('lst', [[True, False, True], [True, True, True], [False, False, False]])
    def test_getitem_boolean_list(self, lst):
        df = DataFrame(np.arange(12).reshape(3, 4))
        result = df[lst]
        expected = df.loc[df.index[lst]]
        tm.assert_frame_equal(result, expected)

    def test_getitem_boolean_iadd(self):
        arr = np.random.default_rng(2).standard_normal((5, 5))
        df = DataFrame(arr.copy(), columns=['A', 'B', 'C', 'D', 'E'])
        df[df < 0] += 1
        arr[arr < 0] += 1
        tm.assert_almost_equal(df.values, arr)

    def test_boolean_index_empty_corner(self):
        blah = DataFrame(np.empty([0, 1]), columns=['A'], index=DatetimeIndex([]))
        k = np.array([], bool)
        blah[k]
        blah[k] = 0

    def test_getitem_ix_mixed_integer(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)), index=[1, 10, 'C', 'E'], columns=[1, 2, 3])
        result = df.iloc[:-1]
        expected = df.loc[df.index[:-1]]
        tm.assert_frame_equal(result, expected)
        result = df.loc[[1, 10]]
        expected = df.loc[Index([1, 10])]
        tm.assert_frame_equal(result, expected)

    def test_getitem_ix_mixed_integer2(self):
        df = DataFrame({'rna': (1.5, 2.2, 3.2, 4.5), -1000: [11, 21, 36, 40], 0: [10, 22, 43, 34], 1000: [0, 10, 20, 30]}, columns=['rna', -1000, 0, 1000])
        result = df[[1000]]
        expected = df.iloc[:, [3]]
        tm.assert_frame_equal(result, expected)
        result = df[[-1000]]
        expected = df.iloc[:, [1]]
        tm.assert_frame_equal(result, expected)

    def test_getattr(self, float_frame):
        tm.assert_series_equal(float_frame.A, float_frame['A'])
        msg = "'DataFrame' object has no attribute 'NONEXISTENT_NAME'"
        with pytest.raises(AttributeError, match=msg):
            float_frame.NONEXISTENT_NAME

    def test_setattr_column(self):
        df = DataFrame({'foobar': 1}, index=range(10))
        df.foobar = 5
        assert (df.foobar == 5).all()

    def test_setitem(self, float_frame, using_copy_on_write, warn_copy_on_write, using_infer_string):
        series = float_frame['A'][::2]
        float_frame['col5'] = series
        assert 'col5' in float_frame
        assert len(series) == 15
        assert len(float_frame) == 30
        exp = np.ravel(np.column_stack((series.values, [np.nan] * 15)))
        exp = Series(exp, index=float_frame.index, name='col5')
        tm.assert_series_equal(float_frame['col5'], exp)
        series = float_frame['A']
        float_frame['col6'] = series
        tm.assert_series_equal(series, float_frame['col6'], check_names=False)
        arr = np.random.default_rng(2).standard_normal(len(float_frame))
        float_frame['col9'] = arr
        assert (float_frame['col9'] == arr).all()
        float_frame['col7'] = 5
        assert (float_frame['col7'] == 5).all()
        float_frame['col0'] = 3.14
        assert (float_frame['col0'] == 3.14).all()
        float_frame['col8'] = 'foo'
        assert (float_frame['col8'] == 'foo').all()
        smaller = float_frame[:2]
        msg = '\\nA value is trying to be set on a copy of a slice from a DataFrame'
        if using_copy_on_write or warn_copy_on_write:
            smaller['col10'] = ['1', '2']
        else:
            with pytest.raises(SettingWithCopyError, match=msg):
                smaller['col10'] = ['1', '2']
        if using_infer_string:
            assert smaller['col10'].dtype == 'string'
        else:
            assert smaller['col10'].dtype == np.object_
        assert (smaller['col10'] == ['1', '2']).all()

    def test_setitem2(self):
        df = DataFrame([[0, 0]])
        df.iloc[0] = np.nan
        expected = DataFrame([[np.nan, np.nan]])
        tm.assert_frame_equal(df, expected)
        df = DataFrame([[0, 0]])
        df.loc[0] = np.nan
        tm.assert_frame_equal(df, expected)

    def test_setitem_boolean(self, float_frame):
        df = float_frame.copy()
        values = float_frame.values.copy()
        df[df['A'] > 0] = 4
        values[values[:, 0] > 0] = 4
        tm.assert_almost_equal(df.values, values)
        series = df['A'] == 4
        series = series.reindex(df.index[::-1])
        df[series] = 1
        values[values[:, 0] == 4] = 1
        tm.assert_almost_equal(df.values, values)
        df[df > 0] = 5
        values[values > 0] = 5
        tm.assert_almost_equal(df.values, values)
        df[df == 5] = 0
        values[values == 5] = 0
        tm.assert_almost_equal(df.values, values)
        df[df[:-1] < 0] = 2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        tm.assert_almost_equal(df.values, values)
        df[df[::-1] == 2] = 3
        values[values == 2] = 3
        tm.assert_almost_equal(df.values, values)
        msg = 'Must pass DataFrame or 2-d ndarray with boolean values only'
        with pytest.raises(TypeError, match=msg):
            df[df * 0] = 2
        df_orig = df.copy()
        mask = df > np.abs(df)
        df[df > np.abs(df)] = np.nan
        values = df_orig.values.copy()
        values[mask.values] = np.nan
        expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
        tm.assert_frame_equal(df, expected)
        df[df > np.abs(df)] = df * 2
        np.putmask(values, mask.values, df.values * 2)
        expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
        tm.assert_frame_equal(df, expected)

    def test_setitem_cast(self, float_frame):
        float_frame['D'] = float_frame['D'].astype('i8')
        assert float_frame['D'].dtype == np.int64
        float_frame['B'] = 0
        assert float_frame['B'].dtype == np.int64
        float_frame['B'] = np.arange(len(float_frame))
        assert issubclass(float_frame['B'].dtype.type, np.integer)
        float_frame['foo'] = 'bar'
        float_frame['foo'] = 0
        assert float_frame['foo'].dtype == np.int64
        float_frame['foo'] = 'bar'
        float_frame['foo'] = 2.5
        assert float_frame['foo'].dtype == np.float64
        float_frame['something'] = 0
        assert float_frame['something'].dtype == np.int64
        float_frame['something'] = 2
        assert float_frame['something'].dtype == np.int64
        float_frame['something'] = 2.5
        assert float_frame['something'].dtype == np.float64

    def test_setitem_corner(self, float_frame, using_infer_string):
        df = DataFrame({'B': [1.0, 2.0, 3.0], 'C': ['a', 'b', 'c']}, index=np.arange(3))
        del df['B']
        df['B'] = [1.0, 2.0, 3.0]
        assert 'B' in df
        assert len(df.columns) == 2
        df['A'] = 'beginning'
        df['E'] = 'foo'
        df['D'] = 'bar'
        df[datetime.now()] = 'date'
        df[datetime.now()] = 5.0
        dm = DataFrame(index=float_frame.index)
        dm['A'] = 'foo'
        dm['B'] = 'bar'
        assert len(dm.columns) == 2
        assert dm.values.dtype == np.object_
        dm['C'] = 1
        assert dm['C'].dtype == np.int64
        dm['E'] = 1.0
        assert dm['E'].dtype == np.float64
        dm['A'] = 'bar'
        assert 'bar' == dm['A'].iloc[0]
        dm = DataFrame(index=np.arange(3))
        dm['A'] = 1
        dm['foo'] = 'bar'
        del dm['foo']
        dm['foo'] = 'bar'
        if using_infer_string:
            assert dm['foo'].dtype == 'string'
        else:
            assert dm['foo'].dtype == np.object_
        dm['coercible'] = ['1', '2', '3']
        if using_infer_string:
            assert dm['coercible'].dtype == 'string'
        else:
            assert dm['coercible'].dtype == np.object_

    def test_setitem_corner2(self):
        data = {'title': ['foobar', 'bar', 'foobar'] + ['foobar'] * 17, 'cruft': np.random.default_rng(2).random(20)}
        df = DataFrame(data)
        ix = df[df['title'] == 'bar'].index
        df.loc[ix, ['title']] = 'foobar'
        df.loc[ix, ['cruft']] = 0
        assert df.loc[1, 'title'] == 'foobar'
        assert df.loc[1, 'cruft'] == 0

    def test_setitem_ambig(self, using_infer_string):
        dm = DataFrame(index=range(3), columns=range(3))
        coercable_series = Series([Decimal(1) for _ in range(3)], index=range(3))
        uncoercable_series = Series(['foo', 'bzr', 'baz'], index=range(3))
        dm[0] = np.ones(3)
        assert len(dm.columns) == 3
        dm[1] = coercable_series
        assert len(dm.columns) == 3
        dm[2] = uncoercable_series
        assert len(dm.columns) == 3
        if using_infer_string:
            assert dm[2].dtype == 'string'
        else:
            assert dm[2].dtype == np.object_

    def test_setitem_None(self, float_frame, using_infer_string):
        float_frame[None] = float_frame['A']
        key = None if not using_infer_string else np.nan
        tm.assert_series_equal(float_frame.iloc[:, -1], float_frame['A'], check_names=False)
        tm.assert_series_equal(float_frame.loc[:, key], float_frame['A'], check_names=False)
        tm.assert_series_equal(float_frame[key], float_frame['A'], check_names=False)

    def test_loc_setitem_boolean_mask_allfalse(self):
        df = DataFrame({'a': ['1', '2', '3'], 'b': ['11', '22', '33'], 'c': ['111', '222', '333']})
        result = df.copy()
        result.loc[result.b.isna(), 'a'] = result.a.copy()
        tm.assert_frame_equal(result, df)

    def test_getitem_fancy_slice_integers_step(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        df.iloc[:8:2]
        df.iloc[:8:2] = np.nan
        assert isna(df.iloc[:8:2]).values.all()

    def test_getitem_setitem_integer_slice_keyerrors(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), index=range(0, 20, 2))
        cp = df.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).values.all()
        cp = df.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()
        result = df.iloc[2:6]
        result2 = df.loc[3:11]
        expected = df.reindex([4, 6, 8, 10])
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)
        df2 = df.iloc[list(range(5)) + list(range(5, 10))[::-1]]
        with pytest.raises(KeyError, match='^3$'):
            df2.loc[3:11]
        with pytest.raises(KeyError, match='^3$'):
            df2.loc[3:11] = 0

    @td.skip_array_manager_invalid_test
    def test_fancy_getitem_slice_mixed(self, float_frame, float_string_frame, using_copy_on_write, warn_copy_on_write):
        sliced = float_string_frame.iloc[:, -3:]
        assert sliced['D'].dtype == np.float64
        original = float_frame.copy()
        sliced = float_frame.iloc[:, -3:]
        assert np.shares_memory(sliced['C']._values, float_frame['C']._values)
        with tm.assert_cow_warning(warn_copy_on_write):
            sliced.loc[:, 'C'] = 4.0
        if not using_copy_on_write:
            assert (float_frame['C'] == 4).all()
            np.shares_memory(sliced['C']._values, float_frame['C']._values)
        else:
            tm.assert_frame_equal(float_frame, original)

    def test_getitem_setitem_non_ix_labels(self):
        df = DataFrame(range(20), index=date_range('2020-01-01', periods=20))
        start, end = df.index[[5, 10]]
        result = df.loc[start:end]
        result2 = df[start:end]
        expected = df[5:11]
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)
        result = df.copy()
        result.loc[start:end] = 0
        result2 = df.copy()
        result2[start:end] = 0
        expected = df.copy()
        expected[5:11] = 0
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_ix_multi_take(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        rs = df.loc[df.index == 0, :]
        xp = df.reindex([0])
        tm.assert_frame_equal(rs, xp)
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        rs = df.loc[df.index == 0, df.columns == 1]
        xp = df.reindex(index=[0], columns=[1])
        tm.assert_frame_equal(rs, xp)

    def test_getitem_fancy_scalar(self, float_frame):
        f = float_frame
        ix = f.loc
        for col in f.columns:
            ts = f[col]
            for idx in f.index[::5]:
                assert ix[idx, col] == ts[idx]

    @td.skip_array_manager_invalid_test
    def test_setitem_fancy_scalar(self, float_frame):
        f = float_frame
        expected = float_frame.copy()
        ix = f.loc
        for j, col in enumerate(f.columns):
            f[col]
            for idx in f.index[::5]:
                i = f.index.get_loc(idx)
                val = np.random.default_rng(2).standard_normal()
                expected.iloc[i, j] = val
                ix[idx, col] = val
                tm.assert_frame_equal(f, expected)

    def test_getitem_fancy_boolean(self, float_frame):
        f = float_frame
        ix = f.loc
        expected = f.reindex(columns=['B', 'D'])
        result = ix[:, [False, True, False, True]]
        tm.assert_frame_equal(result, expected)
        expected = f.reindex(index=f.index[5:10], columns=['B', 'D'])
        result = ix[f.index[5:10], [False, True, False, True]]
        tm.assert_frame_equal(result, expected)
        boolvec = f.index > f.index[7]
        expected = f.reindex(index=f.index[boolvec])
        result = ix[boolvec]
        tm.assert_frame_equal(result, expected)
        result = ix[boolvec, :]
        tm.assert_frame_equal(result, expected)
        result = ix[boolvec, f.columns[2:]]
        expected = f.reindex(index=f.index[boolvec], columns=['C', 'D'])
        tm.assert_frame_equal(result, expected)

    @td.skip_array_manager_invalid_test
    def test_setitem_fancy_boolean(self, float_frame):
        frame = float_frame.copy()
        expected = float_frame.copy()
        values = expected.values.copy()
        mask = frame['A'] > 0
        frame.loc[mask] = 0.0
        values[mask.values] = 0.0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(frame, expected)
        frame = float_frame.copy()
        expected = float_frame.copy()
        values = expected.values.copy()
        frame.loc[mask, ['A', 'B']] = 0.0
        values[mask.values, :2] = 0.0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(frame, expected)

    def test_getitem_fancy_ints(self, float_frame):
        result = float_frame.iloc[[1, 4, 7]]
        expected = float_frame.loc[float_frame.index[[1, 4, 7]]]
        tm.assert_frame_equal(result, expected)
        result = float_frame.iloc[:, [2, 0, 1]]
        expected = float_frame.loc[:, float_frame.columns[[2, 0, 1]]]
        tm.assert_frame_equal(result, expected)

    def test_getitem_setitem_boolean_misaligned(self, float_frame):
        mask = float_frame['A'][::-1] > 1
        result = float_frame.loc[mask]
        expected = float_frame.loc[mask[::-1]]
        tm.assert_frame_equal(result, expected)
        cp = float_frame.copy()
        expected = float_frame.copy()
        cp.loc[mask] = 0
        expected.loc[mask] = 0
        tm.assert_frame_equal(cp, expected)

    def test_getitem_setitem_boolean_multi(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        k1 = np.array([True, False, True])
        k2 = np.array([False, True])
        result = df.loc[k1, k2]
        expected = df.loc[[0, 2], [1]]
        tm.assert_frame_equal(result, expected)
        expected = df.copy()
        df.loc[np.array([True, False, True]), np.array([False, True])] = 5
        expected.loc[[0, 2], [1]] = 5
        tm.assert_frame_equal(df, expected)

    def test_getitem_setitem_float_labels(self, using_array_manager):
        index = Index([1.5, 2, 3, 4, 5])
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)
        result = df.loc[1.5:4]
        expected = df.reindex([1.5, 2, 3, 4])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 4
        result = df.loc[4:5]
        expected = df.reindex([4, 5])
        tm.assert_frame_equal(result, expected, check_index_type=False)
        assert len(result) == 2
        result = df.loc[4:5]
        expected = df.reindex([4.0, 5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 2
        result = df.loc[1:2]
        expected = df.iloc[0:2]
        tm.assert_frame_equal(result, expected)
        expected = df.iloc[0:2]
        msg = 'The behavior of obj\\[i:j\\] with a float-dtype index'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df[1:2]
        tm.assert_frame_equal(result, expected)
        index = Index([1.0, 2.5, 3.5, 4.5, 5.0])
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)
        msg = 'cannot do positional indexing on Index with these indexers \\[1.0\\] of type float'
        with pytest.raises(TypeError, match=msg):
            df.iloc[1.0:5]
        result = df.iloc[4:5]
        expected = df.reindex([5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 1
        cp = df.copy()
        with pytest.raises(TypeError, match=_slice_msg):
            cp.iloc[1.0:5] = 0
        with pytest.raises(TypeError, match=msg):
            result = cp.iloc[1.0:5] == 0
        assert result.values.all()
        assert (cp.iloc[0:1] == df.iloc[0:1]).values.all()
        cp = df.copy()
        cp.iloc[4:5] = 0
        assert (cp.iloc[4:5] == 0).values.all()
        assert (cp.iloc[0:4] == df.iloc[0:4]).values.all()
        result = df.loc[1.0:5]
        expected = df
        tm.assert_frame_equal(result, expected)
        assert len(result) == 5
        result = df.loc[1.1:5]
        expected = df.reindex([2.5, 3.5, 4.5, 5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 4
        result = df.loc[4.51:5]
        expected = df.reindex([5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 1
        result = df.loc[1.0:5.0]
        expected = df.reindex([1.0, 2.5, 3.5, 4.5, 5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 5
        cp = df.copy()
        cp.loc[1.0:5.0] = 0
        result = cp.loc[1.0:5.0]
        assert (result == 0).values.all()

    def test_setitem_single_column_mixed_datetime(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=['a', 'b', 'c', 'd', 'e'], columns=['foo', 'bar', 'baz'])
        df['timestamp'] = Timestamp('20010102')
        result = df.dtypes
        expected = Series([np.dtype('float64')] * 3 + [np.dtype('datetime64[s]')], index=['foo', 'bar', 'baz', 'timestamp'])
        tm.assert_series_equal(result, expected)
        with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
            df.loc['b', 'timestamp'] = iNaT
        assert not isna(df.loc['b', 'timestamp'])
        assert df['timestamp'].dtype == np.object_
        assert df.loc['b', 'timestamp'] == iNaT
        df.loc['c', 'timestamp'] = np.nan
        assert isna(df.loc['c', 'timestamp'])
        df.loc['d', :] = np.nan
        assert not isna(df.loc['c', :]).all()

    def test_setitem_mixed_datetime(self):
        expected = DataFrame({'a': [0, 0, 0, 0, 13, 14], 'b': [datetime(2012, 1, 1), 1, 'x', 'y', datetime(2013, 1, 1), datetime(2014, 1, 1)]})
        df = DataFrame(0, columns=list('ab'), index=range(6))
        df['b'] = pd.NaT
        df.loc[0, 'b'] = datetime(2012, 1, 1)
        with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
            df.loc[1, 'b'] = 1
        df.loc[[2, 3], 'b'] = ('x', 'y')
        A = np.array([[13, np.datetime64('2013-01-01T00:00:00')], [14, np.datetime64('2014-01-01T00:00:00')]])
        df.loc[[4, 5], ['a', 'b']] = A
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_float(self, float_frame):
        piece = float_frame.loc[float_frame.index[:2], ['A', 'B']]
        float_frame.loc[float_frame.index[-2]:, ['A', 'B']] = piece.values
        result = float_frame.loc[float_frame.index[-2:], ['A', 'B']].values
        expected = piece.values
        tm.assert_almost_equal(result, expected)

    def test_setitem_frame_mixed(self, float_string_frame):
        f = float_string_frame.copy()
        piece = DataFrame([[1.0, 2.0], [3.0, 4.0]], index=f.index[0:2], columns=['A', 'B'])
        key = (f.index[slice(None, 2)], ['A', 'B'])
        f.loc[key] = piece
        tm.assert_almost_equal(f.loc[f.index[0:2], ['A', 'B']].values, piece.values)

    def test_setitem_frame_mixed_rows_unaligned(self, float_string_frame):
        f = float_string_frame.copy()
        piece = DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], index=list(f.index[0:2]) + ['foo', 'bar'], columns=['A', 'B'])
        key = (f.index[slice(None, 2)], ['A', 'B'])
        f.loc[key] = piece
        tm.assert_almost_equal(f.loc[f.index[0:2], ['A', 'B']].values, piece.values[0:2])

    def test_setitem_frame_mixed_key_unaligned(self, float_string_frame):
        f = float_string_frame.copy()
        piece = f.loc[f.index[:2], ['A']]
        piece.index = f.index[-2:]
        key = (f.index[slice(-2, None)], ['A', 'B'])
        f.loc[key] = piece
        piece['B'] = np.nan
        tm.assert_almost_equal(f.loc[f.index[-2:], ['A', 'B']].values, piece.values)

    def test_setitem_frame_mixed_ndarray(self, float_string_frame):
        f = float_string_frame.copy()
        piece = float_string_frame.loc[f.index[:2], ['A', 'B']]
        key = (f.index[slice(-2, None)], ['A', 'B'])
        f.loc[key] = piece.values
        tm.assert_almost_equal(f.loc[f.index[-2:], ['A', 'B']].values, piece.values)

    def test_setitem_frame_upcast(self):
        df = DataFrame([[1, 2, 'foo'], [3, 4, 'bar']], columns=['A', 'B', 'C'])
        df2 = df.copy()
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            df2.loc[:, ['A', 'B']] = df.loc[:, ['A', 'B']] + 0.5
        expected = df.reindex(columns=['A', 'B'])
        expected += 0.5
        expected['C'] = df['C']
        tm.assert_frame_equal(df2, expected)

    def test_setitem_frame_align(self, float_frame):
        piece = float_frame.loc[float_frame.index[:2], ['A', 'B']]
        piece.index = float_frame.index[-2:]
        piece.columns = ['A', 'B']
        float_frame.loc[float_frame.index[-2:], ['A', 'B']] = piece
        result = float_frame.loc[float_frame.index[-2:], ['A', 'B']].values
        expected = piece.values
        tm.assert_almost_equal(result, expected)

    def test_getitem_setitem_ix_duplicates(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=['foo', 'foo', 'bar', 'baz', 'bar'])
        result = df.loc['foo']
        expected = df[:2]
        tm.assert_frame_equal(result, expected)
        result = df.loc['bar']
        expected = df.iloc[[2, 4]]
        tm.assert_frame_equal(result, expected)
        result = df.loc['baz']
        expected = df.iloc[3]
        tm.assert_series_equal(result, expected)

    def test_getitem_ix_boolean_duplicates_multiple(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=['foo', 'foo', 'bar', 'baz', 'bar'])
        result = df.loc[['bar']]
        exp = df.iloc[[2, 4]]
        tm.assert_frame_equal(result, exp)
        result = df.loc[df[1] > 0]
        exp = df[df[1] > 0]
        tm.assert_frame_equal(result, exp)
        result = df.loc[df[0] > 0]
        exp = df[df[0] > 0]
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize('bool_value', [True, False])
    def test_getitem_setitem_ix_bool_keyerror(self, bool_value):
        df = DataFrame({'a': [1, 2, 3]})
        message = f'{bool_value}: boolean label can not be used without a boolean index'
        with pytest.raises(KeyError, match=message):
            df.loc[bool_value]
        msg = 'cannot use a single bool to index into setitem'
        with pytest.raises(KeyError, match=msg):
            df.loc[bool_value] = 0

    def test_single_element_ix_dont_upcast(self, float_frame):
        float_frame['E'] = 1
        assert issubclass(float_frame['E'].dtype.type, (int, np.integer))
        result = float_frame.loc[float_frame.index[5], 'E']
        assert is_integer(result)
        df = DataFrame({'a': [1.23]})
        df['b'] = 666
        result = df.loc[0, 'b']
        assert is_integer(result)
        expected = Series([666], [0], name='b')
        result = df.loc[[0], 'b']
        tm.assert_series_equal(result, expected)

    def test_iloc_callable_tuple_return_value(self):
        df = DataFrame(np.arange(40).reshape(10, 4), index=range(0, 20, 2))
        msg = 'callable with iloc'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.iloc[lambda _: (0,)]
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.iloc[lambda _: (0,)] = 1

    def test_iloc_row(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=range(0, 20, 2))
        result = df.iloc[1]
        exp = df.loc[2]
        tm.assert_series_equal(result, exp)
        result = df.iloc[2]
        exp = df.loc[4]
        tm.assert_series_equal(result, exp)
        result = df.iloc[slice(4, 8)]
        expected = df.loc[8:14]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[[1, 2, 4, 6]]
        expected = df.reindex(df.index[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_row_slice_view(self, using_copy_on_write, warn_copy_on_write):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=range(0, 20, 2))
        original = df.copy()
        subset = df.iloc[slice(4, 8)]
        assert np.shares_memory(df[2], subset[2])
        exp_col = original[2].copy()
        with tm.assert_cow_warning(warn_copy_on_write):
            subset.loc[:, 2] = 0.0
        if not using_copy_on_write:
            exp_col._values[4:8] = 0.0
            assert np.shares_memory(df[2], subset[2])
        tm.assert_series_equal(df[2], exp_col)

    def test_iloc_col(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2))
        result = df.iloc[:, 1]
        exp = df.loc[:, 2]
        tm.assert_series_equal(result, exp)
        result = df.iloc[:, 2]
        exp = df.loc[:, 4]
        tm.assert_series_equal(result, exp)
        result = df.iloc[:, slice(4, 8)]
        expected = df.loc[:, 8:14]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[:, [1, 2, 4, 6]]
        expected = df.reindex(columns=df.columns[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_col_slice_view(self, using_array_manager, using_copy_on_write, warn_copy_on_write):
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2))
        original = df.copy()
        subset = df.iloc[:, slice(4, 8)]
        if not using_array_manager and (not using_copy_on_write):
            assert np.shares_memory(df[8]._values, subset[8]._values)
            with tm.assert_cow_warning(warn_copy_on_write):
                subset.loc[:, 8] = 0.0
            assert (df[8] == 0).all()
            assert np.shares_memory(df[8]._values, subset[8]._values)
        else:
            if using_copy_on_write:
                assert np.shares_memory(df[8]._values, subset[8]._values)
            subset[8] = 0.0
            assert (subset[8] == 0).all()
            tm.assert_frame_equal(df, original)

    def test_loc_duplicates(self):
        trange = date_range(start=Timestamp(year=2017, month=1, day=1), end=Timestamp(year=2017, month=1, day=5))
        trange = trange.insert(loc=5, item=Timestamp(year=2017, month=1, day=5))
        df = DataFrame(0, index=trange, columns=['A', 'B'])
        bool_idx = np.array([False, False, False, False, False, True])
        df.loc[trange[bool_idx], 'A'] = 6
        expected = DataFrame({'A': [0, 0, 0, 0, 6, 6], 'B': [0, 0, 0, 0, 0, 0]}, index=trange)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(0, index=trange, columns=['A', 'B'])
        df.loc[trange[bool_idx], 'A'] += 6
        tm.assert_frame_equal(df, expected)

    def test_setitem_with_unaligned_tz_aware_datetime_column(self):
        column = Series(date_range('2015-01-01', periods=3, tz='utc'), name='dates')
        df = DataFrame({'dates': column})
        df['dates'] = column[[1, 0, 2]]
        tm.assert_series_equal(df['dates'], column)
        df = DataFrame({'dates': column})
        df.loc[[0, 1, 2], 'dates'] = column[[1, 0, 2]]
        tm.assert_series_equal(df['dates'], column)

    def test_loc_setitem_datetimelike_with_inference(self):
        one_hour = timedelta(hours=1)
        df = DataFrame(index=date_range('20130101', periods=4))
        df['A'] = np.array([1 * one_hour] * 4, dtype='m8[ns]')
        df.loc[:, 'B'] = np.array([2 * one_hour] * 4, dtype='m8[ns]')
        df.loc[df.index[:3], 'C'] = np.array([3 * one_hour] * 3, dtype='m8[ns]')
        df.loc[:, 'D'] = np.array([4 * one_hour] * 4, dtype='m8[ns]')
        df.loc[df.index[:3], 'E'] = np.array([5 * one_hour] * 3, dtype='m8[ns]')
        df['F'] = np.timedelta64('NaT')
        df.loc[df.index[:-1], 'F'] = np.array([6 * one_hour] * 3, dtype='m8[ns]')
        df.loc[df.index[-3]:, 'G'] = date_range('20130101', periods=3)
        df['H'] = np.datetime64('NaT')
        result = df.dtypes
        expected = Series([np.dtype('timedelta64[ns]')] * 6 + [np.dtype('datetime64[ns]')] * 2, index=list('ABCDEFGH'))
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean_indexing_mixed(self):
        df = DataFrame({0: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan}, 1: {35: np.nan, 40: 0.326323168594462, 43: np.nan, 49: 0.326323168594462, 50: 0.3911472448057814}, 2: {35: np.nan, 40: np.nan, 43: 0.29012581014105987, 49: np.nan, 50: np.nan}, 3: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan}, 4: {35: 0.34215328467153283, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan}, 'y': {35: 0, 40: 0, 43: 0, 49: 0, 50: 1}})
        df2 = df.copy()
        df2[df2 > 0.3] = 1
        expected = df.copy()
        expected.loc[40, 1] = 1
        expected.loc[49, 1] = 1
        expected.loc[50, 1] = 1
        expected.loc[35, 4] = 1
        tm.assert_frame_equal(df2, expected)
        df['foo'] = 'test'
        msg = 'not supported between instances|unorderable types'
        with pytest.raises(TypeError, match=msg):
            df[df > 0.3] = 1

    def test_type_error_multiindex(self):
        mi = MultiIndex.from_product([['x', 'y'], [0, 1]], names=[None, 'c'])
        dg = DataFrame([[1, 1, 2, 2], [3, 3, 4, 4]], columns=mi, index=Index([0, 1], name='i'))
        with pytest.raises(InvalidIndexError, match='slice'):
            dg[:, 0]
        index = Index(range(2), name='i')
        columns = MultiIndex(levels=[['x', 'y'], [0, 1]], codes=[[0, 1], [0, 0]], names=[None, 'c'])
        expected = DataFrame([[1, 2], [3, 4]], columns=columns, index=index)
        result = dg.loc[:, (slice(None), 0)]
        tm.assert_frame_equal(result, expected)
        name = ('x', 0)
        index = Index(range(2), name='i')
        expected = Series([1, 3], index=index, name=name)
        result = dg['x', 0]
        tm.assert_series_equal(result, expected)

    def test_getitem_interval_index_partial_indexing(self):
        df = DataFrame(np.ones((3, 4)), columns=pd.IntervalIndex.from_breaks(np.arange(5)))
        expected = df.iloc[:, 0]
        res = df[0.5]
        tm.assert_series_equal(res, expected)
        res = df.loc[:, 0.5]
        tm.assert_series_equal(res, expected)

    def test_setitem_array_as_cell_value(self):
        df = DataFrame(columns=['a', 'b'], dtype=object)
        df.loc[0] = {'a': np.zeros((2,)), 'b': np.zeros((2, 2))}
        expected = DataFrame({'a': [np.zeros((2,))], 'b': [np.zeros((2, 2))]})
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_nullable_2d_values(self):
        df = DataFrame({'A': [1, 2, 3]}, dtype='Int64')
        orig = df.copy()
        df.loc[:] = df.values[:, ::-1]
        tm.assert_frame_equal(df, orig)
        df.loc[:] = pd.core.arrays.NumpyExtensionArray(df.values[:, ::-1])
        tm.assert_frame_equal(df, orig)
        df.iloc[:] = df.iloc[:, :].copy()
        tm.assert_frame_equal(df, orig)

    def test_getitem_segfault_with_empty_like_object(self):
        df = DataFrame(np.empty((1, 1), dtype=object))
        df[0] = np.empty_like(df[0])
        df[[0]]

    @pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
    @pytest.mark.parametrize('null', [pd.NaT, pd.NaT.to_numpy('M8[ns]'), pd.NaT.to_numpy('m8[ns]')])
    def test_setting_mismatched_na_into_nullable_fails(self, null, any_numeric_ea_dtype):
        df = DataFrame({'A': [1, 2, 3]}, dtype=any_numeric_ea_dtype)
        ser = df['A'].copy()
        arr = ser._values
        msg = '|'.join(['timedelta64\\[ns\\] cannot be converted to (Floating|Integer)Dtype', 'datetime64\\[ns\\] cannot be converted to (Floating|Integer)Dtype', "'values' contains non-numeric NA", "Invalid value '.*' for dtype (U?Int|Float)\\d{1,2}"])
        with pytest.raises(TypeError, match=msg):
            arr[0] = null
        with pytest.raises(TypeError, match=msg):
            arr[:2] = [null, null]
        with pytest.raises(TypeError, match=msg):
            ser[0] = null
        with pytest.raises(TypeError, match=msg):
            ser[:2] = [null, null]
        with pytest.raises(TypeError, match=msg):
            ser.iloc[0] = null
        with pytest.raises(TypeError, match=msg):
            ser.iloc[:2] = [null, null]
        with pytest.raises(TypeError, match=msg):
            df.iloc[0, 0] = null
        with pytest.raises(TypeError, match=msg):
            df.iloc[:2, 0] = [null, null]
        df2 = df.copy()
        df2['B'] = ser.copy()
        with pytest.raises(TypeError, match=msg):
            df2.iloc[0, 0] = null
        with pytest.raises(TypeError, match=msg):
            df2.iloc[:2, 0] = [null, null]

    def test_loc_expand_empty_frame_keep_index_name(self):
        df = DataFrame(columns=['b'], index=Index([], name='a'))
        df.loc[0] = 1
        expected = DataFrame({'b': [1]}, index=Index([0], name='a'))
        tm.assert_frame_equal(df, expected)

    def test_loc_expand_empty_frame_keep_midx_names(self):
        df = DataFrame(columns=['d'], index=MultiIndex.from_tuples([], names=['a', 'b', 'c']))
        df.loc[1, 2, 3] = 'foo'
        expected = DataFrame({'d': ['foo']}, index=MultiIndex.from_tuples([(1, 2, 3)], names=['a', 'b', 'c']))
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('val, idxr', [('x', 'a'), ('x', ['a']), (1, 'a'), (1, ['a'])])
    def test_loc_setitem_rhs_frame(self, idxr, val):
        df = DataFrame({'a': [1, 2]})
        with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
            df.loc[:, idxr] = DataFrame({'a': [val, 11]}, index=[1, 2])
        expected = DataFrame({'a': [np.nan, val]})
        tm.assert_frame_equal(df, expected)

    @td.skip_array_manager_invalid_test
    def test_iloc_setitem_enlarge_no_warning(self, warn_copy_on_write):
        df = DataFrame(columns=['a', 'b'])
        expected = df.copy()
        view = df[:]
        df.iloc[:, 0] = np.array([1, 2], dtype=np.float64)
        tm.assert_frame_equal(view, expected)

    def test_loc_internals_not_updated_correctly(self):
        df = DataFrame({'bool_col': True, 'a': 1, 'b': 2.5}, index=MultiIndex.from_arrays([[1, 2], [1, 2]], names=['idx1', 'idx2']))
        idx = [(1, 1)]
        df['c'] = 3
        df.loc[idx, 'c'] = 0
        df.loc[idx, 'c']
        df.loc[idx, ['a', 'b']]
        df.loc[idx, 'c'] = 15
        result = df.loc[idx, 'c']
        expected = df = Series(15, index=MultiIndex.from_arrays([[1], [1]], names=['idx1', 'idx2']), name='c')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('val', [None, [None], pd.NA, [pd.NA]])
    def test_iloc_setitem_string_list_na(self, val):
        df = DataFrame({'a': ['a', 'b', 'c']}, dtype='string')
        df.iloc[[0], :] = val
        expected = DataFrame({'a': [pd.NA, 'b', 'c']}, dtype='string')
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('val', [None, pd.NA])
    def test_iloc_setitem_string_na(self, val):
        df = DataFrame({'a': ['a', 'b', 'c']}, dtype='string')
        df.iloc[0, :] = val
        expected = DataFrame({'a': [pd.NA, 'b', 'c']}, dtype='string')
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('func', [list, Series, np.array])
    def test_iloc_setitem_ea_null_slice_length_one_list(self, func):
        df = DataFrame({'a': [1, 2, 3]}, dtype='Int64')
        df.iloc[:, func([0])] = 5
        expected = DataFrame({'a': [5, 5, 5]}, dtype='Int64')
        tm.assert_frame_equal(df, expected)

    def test_loc_named_tuple_for_midx(self):
        df = DataFrame(index=MultiIndex.from_product([['A', 'B'], ['a', 'b', 'c']], names=['first', 'second']))
        indexer_tuple = namedtuple('Indexer', df.index.names)
        idxr = indexer_tuple(first='A', second=['a', 'b'])
        result = df.loc[idxr, :]
        expected = DataFrame(index=MultiIndex.from_tuples([('A', 'a'), ('A', 'b')], names=['first', 'second']))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('indexer', [['a'], 'a'])
    @pytest.mark.parametrize('col', [{}, {'b': 1}])
    def test_set_2d_casting_date_to_int(self, col, indexer):
        df = DataFrame({'a': [Timestamp('2022-12-29'), Timestamp('2022-12-30')], **col})
        df.loc[[1], indexer] = df['a'] + pd.Timedelta(days=1)
        expected = DataFrame({'a': [Timestamp('2022-12-29'), Timestamp('2022-12-31')], **col})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('col', [{}, {'name': 'a'}])
    def test_loc_setitem_reordering_with_all_true_indexer(self, col):
        n = 17
        df = DataFrame({**col, 'x': range(n), 'y': range(n)})
        expected = df.copy()
        df.loc[n * [True], ['x', 'y']] = df[['x', 'y']]
        tm.assert_frame_equal(df, expected)

    def test_loc_rhs_empty_warning(self):
        df = DataFrame(columns=['a', 'b'])
        expected = df.copy()
        rhs = DataFrame(columns=['a'])
        with tm.assert_produces_warning(None):
            df.loc[:, 'a'] = rhs
        tm.assert_frame_equal(df, expected)

    def test_iloc_ea_series_indexer(self):
        df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        indexer = Series([0, 1], dtype='Int64')
        row_indexer = Series([1], dtype='Int64')
        result = df.iloc[row_indexer, indexer]
        expected = DataFrame([[5, 6]], index=[1])
        tm.assert_frame_equal(result, expected)
        result = df.iloc[row_indexer.values, indexer.values]
        tm.assert_frame_equal(result, expected)

    def test_iloc_ea_series_indexer_with_na(self):
        df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        indexer = Series([0, pd.NA], dtype='Int64')
        msg = 'cannot convert'
        with pytest.raises(ValueError, match=msg):
            df.iloc[:, indexer]
        with pytest.raises(ValueError, match=msg):
            df.iloc[:, indexer.values]

    @pytest.mark.parametrize('indexer', [True, (True,)])
    @pytest.mark.parametrize('dtype', [bool, 'boolean'])
    def test_loc_bool_multiindex(self, dtype, indexer):
        midx = MultiIndex.from_arrays([Series([True, True, False, False], dtype=dtype), Series([True, False, True, False], dtype=dtype)], names=['a', 'b'])
        df = DataFrame({'c': [1, 2, 3, 4]}, index=midx)
        with tm.maybe_produces_warning(PerformanceWarning, isinstance(indexer, tuple)):
            result = df.loc[indexer]
        expected = DataFrame({'c': [1, 2]}, index=Index([True, False], name='b', dtype=dtype))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('utc', [False, True])
    @pytest.mark.parametrize('indexer', ['date', ['date']])
    def test_loc_datetime_assignment_dtype_does_not_change(self, utc, indexer):
        df = DataFrame({'date': to_datetime([datetime(2022, 1, 20), datetime(2022, 1, 22)], utc=utc), 'update': [True, False]})
        expected = df.copy(deep=True)
        update_df = df[df['update']]
        df.loc[df['update'], indexer] = update_df['date']
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer, idx', [(tm.loc, 1), (tm.iloc, 2)])
    def test_setitem_value_coercing_dtypes(self, indexer, idx):
        df = DataFrame([['1', np.nan], ['2', np.nan], ['3', np.nan]], dtype=object)
        rhs = DataFrame([[1, np.nan], [2, np.nan]])
        indexer(df)[:idx, :] = rhs
        expected = DataFrame([[1, np.nan], [2, np.nan], ['3', np.nan]], dtype=object)
        tm.assert_frame_equal(df, expected)