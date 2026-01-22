import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
class TestGetitemListLike:

    def test_getitem_list_missing_key(self):
        df = DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0]})
        df.columns = ['x', 'x', 'z']
        with pytest.raises(KeyError, match="\\['y'\\] not in index"):
            df[['x', 'y', 'z']]

    def test_getitem_list_duplicates(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=list('AABC'))
        df.columns.name = 'foo'
        result = df[['B', 'C']]
        assert result.columns.name == 'foo'
        expected = df.iloc[:, 2:]
        tm.assert_frame_equal(result, expected)

    def test_getitem_dupe_cols(self):
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'a', 'b'])
        msg = '"None of [Index([\'baf\'], dtype='
        with pytest.raises(KeyError, match=re.escape(msg)):
            df[['baf']]

    @pytest.mark.parametrize('idx_type', [list, iter, Index, set, lambda keys: dict(zip(keys, range(len(keys)))), lambda keys: dict(zip(keys, range(len(keys)))).keys()], ids=['list', 'iter', 'Index', 'set', 'dict', 'dict_keys'])
    @pytest.mark.parametrize('levels', [1, 2])
    def test_getitem_listlike(self, idx_type, levels, float_frame):
        if levels == 1:
            frame, missing = (float_frame, 'food')
        else:
            frame = DataFrame(np.random.default_rng(2).standard_normal((8, 3)), columns=Index([('foo', 'bar'), ('baz', 'qux'), ('peek', 'aboo')], name=('sth', 'sth2')))
            missing = ('good', 'food')
        keys = [frame.columns[1], frame.columns[0]]
        idx = idx_type(keys)
        idx_check = list(idx_type(keys))
        if isinstance(idx, (set, dict)):
            with pytest.raises(TypeError, match='as an indexer is not supported'):
                frame[idx]
            return
        else:
            result = frame[idx]
        expected = frame.loc[:, idx_check]
        expected.columns.names = frame.columns.names
        tm.assert_frame_equal(result, expected)
        idx = idx_type(keys + [missing])
        with pytest.raises(KeyError, match='not in index'):
            frame[idx]

    def test_getitem_iloc_generator(self):
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        indexer = (x for x in [1, 2])
        result = df.iloc[indexer]
        expected = DataFrame({'a': [2, 3], 'b': [5, 6]}, index=[1, 2])
        tm.assert_frame_equal(result, expected)

    def test_getitem_iloc_two_dimensional_generator(self):
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        indexer = (x for x in [1, 2])
        result = df.iloc[indexer, 1]
        expected = Series([5, 6], name='b', index=[1, 2])
        tm.assert_series_equal(result, expected)

    def test_getitem_iloc_dateoffset_days(self):
        df = DataFrame(list(range(10)), index=date_range('01-01-2022', periods=10, freq=DateOffset(days=1)))
        result = df.loc['2022-01-01':'2022-01-03']
        expected = DataFrame([0, 1, 2], index=DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03'], dtype='datetime64[ns]', freq=DateOffset(days=1)))
        tm.assert_frame_equal(result, expected)
        df = DataFrame(list(range(10)), index=date_range('01-01-2022', periods=10, freq=DateOffset(days=1, hours=2)))
        result = df.loc['2022-01-01':'2022-01-03']
        expected = DataFrame([0, 1, 2], index=DatetimeIndex(['2022-01-01 00:00:00', '2022-01-02 02:00:00', '2022-01-03 04:00:00'], dtype='datetime64[ns]', freq=DateOffset(days=1, hours=2)))
        tm.assert_frame_equal(result, expected)
        df = DataFrame(list(range(10)), index=date_range('01-01-2022', periods=10, freq=DateOffset(minutes=3)))
        result = df.loc['2022-01-01':'2022-01-03']
        tm.assert_frame_equal(result, df)