import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
class TestXS:

    def test_xs(self, float_frame, datetime_frame, using_copy_on_write, warn_copy_on_write):
        float_frame_orig = float_frame.copy()
        idx = float_frame.index[5]
        xs = float_frame.xs(idx)
        for item, value in xs.items():
            if np.isnan(value):
                assert np.isnan(float_frame[item][idx])
            else:
                assert value == float_frame[item][idx]
        test_data = {'A': {'1': 1, '2': 2}, 'B': {'1': '1', '2': '2', '3': '3'}}
        frame = DataFrame(test_data)
        xs = frame.xs('1')
        assert xs.dtype == np.object_
        assert xs['A'] == 1
        assert xs['B'] == '1'
        with pytest.raises(KeyError, match=re.escape("Timestamp('1999-12-31 00:00:00')")):
            datetime_frame.xs(datetime_frame.index[0] - BDay())
        series = float_frame.xs('A', axis=1)
        expected = float_frame['A']
        tm.assert_series_equal(series, expected)
        series = float_frame.xs('A', axis=1)
        with tm.assert_cow_warning(warn_copy_on_write):
            series[:] = 5
        if using_copy_on_write:
            tm.assert_series_equal(float_frame['A'], float_frame_orig['A'])
            assert not (expected == 5).all()
        else:
            assert (expected == 5).all()

    def test_xs_corner(self):
        df = DataFrame(index=[0])
        df['A'] = 1.0
        df['B'] = 'foo'
        df['C'] = 2.0
        df['D'] = 'bar'
        df['E'] = 3.0
        xs = df.xs(0)
        exp = Series([1.0, 'foo', 2.0, 'bar', 3.0], index=list('ABCDE'), name=0)
        tm.assert_series_equal(xs, exp)
        df = DataFrame(index=['a', 'b', 'c'])
        result = df.xs('a')
        expected = Series([], name='a', dtype=np.float64)
        tm.assert_series_equal(result, expected)

    def test_xs_duplicates(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=['b', 'b', 'c', 'b', 'a'])
        cross = df.xs('c')
        exp = df.iloc[2]
        tm.assert_series_equal(cross, exp)

    def test_xs_keep_level(self):
        df = DataFrame({'day': {0: 'sat', 1: 'sun'}, 'flavour': {0: 'strawberry', 1: 'strawberry'}, 'sales': {0: 10, 1: 12}, 'year': {0: 2008, 1: 2008}}).set_index(['year', 'flavour', 'day'])
        result = df.xs('sat', level='day', drop_level=False)
        expected = df[:1]
        tm.assert_frame_equal(result, expected)
        result = df.xs((2008, 'sat'), level=['year', 'day'], drop_level=False)
        tm.assert_frame_equal(result, expected)

    def test_xs_view(self, using_array_manager, using_copy_on_write, warn_copy_on_write):
        dm = DataFrame(np.arange(20.0).reshape(4, 5), index=range(4), columns=range(5))
        df_orig = dm.copy()
        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                dm.xs(2)[:] = 20
            tm.assert_frame_equal(dm, df_orig)
        elif using_array_manager:
            msg = '\\nA value is trying to be set on a copy of a slice from a DataFrame'
            with pytest.raises(SettingWithCopyError, match=msg):
                dm.xs(2)[:] = 20
            assert not (dm.xs(2) == 20).any()
        else:
            with tm.raises_chained_assignment_error():
                dm.xs(2)[:] = 20
            assert (dm.xs(2) == 20).all()