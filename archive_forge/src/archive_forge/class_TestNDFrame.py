from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
class TestNDFrame:

    @pytest.mark.parametrize('ser', [Series(range(10), dtype=np.float64), Series([str(i) for i in range(10)], dtype=object)])
    def test_squeeze_series_noop(self, ser):
        tm.assert_series_equal(ser.squeeze(), ser)

    def test_squeeze_frame_noop(self):
        df = DataFrame(np.eye(2))
        tm.assert_frame_equal(df.squeeze(), df)

    def test_squeeze_frame_reindex(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B')).reindex(columns=['A'])
        tm.assert_series_equal(df.squeeze(), df['A'])

    def test_squeeze_0_len_dim(self):
        empty_series = Series([], name='five', dtype=np.float64)
        empty_frame = DataFrame([empty_series])
        tm.assert_series_equal(empty_series, empty_series.squeeze())
        tm.assert_series_equal(empty_series, empty_frame.squeeze())

    def test_squeeze_axis(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=1, freq='B')).iloc[:, :1]
        assert df.shape == (1, 1)
        tm.assert_series_equal(df.squeeze(axis=0), df.iloc[0])
        tm.assert_series_equal(df.squeeze(axis='index'), df.iloc[0])
        tm.assert_series_equal(df.squeeze(axis=1), df.iloc[:, 0])
        tm.assert_series_equal(df.squeeze(axis='columns'), df.iloc[:, 0])
        assert df.squeeze() == df.iloc[0, 0]
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis=2)
        msg = 'No axis named x for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis='x')

    def test_squeeze_axis_len_3(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=3, freq='B'))
        tm.assert_frame_equal(df.squeeze(axis=0), df)

    def test_numpy_squeeze(self):
        s = Series(range(2), dtype=np.float64)
        tm.assert_series_equal(np.squeeze(s), s)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B')).reindex(columns=['A'])
        tm.assert_series_equal(np.squeeze(df), df['A'])

    @pytest.mark.parametrize('ser', [Series(range(10), dtype=np.float64), Series([str(i) for i in range(10)], dtype=object)])
    def test_transpose_series(self, ser):
        tm.assert_series_equal(ser.transpose(), ser)

    def test_transpose_frame(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        tm.assert_frame_equal(df.transpose().transpose(), df)

    def test_numpy_transpose(self, frame_or_series):
        obj = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        obj = tm.get_obj(obj, frame_or_series)
        if frame_or_series is Series:
            tm.assert_series_equal(np.transpose(obj), obj)
        tm.assert_equal(np.transpose(np.transpose(obj)), obj)
        msg = "the 'axes' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.transpose(obj, axes=1)

    @pytest.mark.parametrize('ser', [Series(range(10), dtype=np.float64), Series([str(i) for i in range(10)], dtype=object)])
    def test_take_series(self, ser):
        indices = [1, 5, -2, 6, 3, -1]
        out = ser.take(indices)
        expected = Series(data=ser.values.take(indices), index=ser.index.take(indices), dtype=ser.dtype)
        tm.assert_series_equal(out, expected)

    def test_take_frame(self):
        indices = [1, 5, -2, 6, 3, -1]
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        out = df.take(indices)
        expected = DataFrame(data=df.values.take(indices, axis=0), index=df.index.take(indices), columns=df.columns)
        tm.assert_frame_equal(out, expected)

    def test_take_invalid_kwargs(self, frame_or_series):
        indices = [-3, 2, 0, 1]
        obj = DataFrame(range(5))
        obj = tm.get_obj(obj, frame_or_series)
        msg = "take\\(\\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            obj.take(indices, foo=2)
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, out=indices)
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, mode='clip')

    def test_axis_classmethods(self, frame_or_series):
        box = frame_or_series
        obj = box(dtype=object)
        values = box._AXIS_TO_AXIS_NUMBER.keys()
        for v in values:
            assert obj._get_axis_number(v) == box._get_axis_number(v)
            assert obj._get_axis_name(v) == box._get_axis_name(v)
            assert obj._get_block_manager_axis(v) == box._get_block_manager_axis(v)

    def test_flags_identity(self, frame_or_series):
        obj = Series([1, 2])
        if frame_or_series is DataFrame:
            obj = obj.to_frame()
        assert obj.flags is obj.flags
        obj2 = obj.copy()
        assert obj2.flags is not obj.flags

    def test_bool_dep(self) -> None:
        msg_warn = 'DataFrame.bool is now deprecated and will be removed in future version of pandas'
        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            DataFrame({'col': [False]}).bool()