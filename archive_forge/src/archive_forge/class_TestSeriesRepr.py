from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestSeriesRepr:

    def test_multilevel_name_print(self, lexsorted_two_level_string_multiindex):
        index = lexsorted_two_level_string_multiindex
        ser = Series(range(len(index)), index=index, name='sth')
        expected = ['first  second', 'foo    one       0', '       two       1', '       three     2', 'bar    one       3', '       two       4', 'baz    two       5', '       three     6', 'qux    one       7', '       two       8', '       three     9', 'Name: sth, dtype: int64']
        expected = '\n'.join(expected)
        assert repr(ser) == expected

    def test_small_name_printing(self):
        s = Series([0, 1, 2])
        s.name = 'test'
        assert 'Name: test' in repr(s)
        s.name = None
        assert 'Name:' not in repr(s)

    def test_big_name_printing(self):
        s = Series(range(1000))
        s.name = 'test'
        assert 'Name: test' in repr(s)
        s.name = None
        assert 'Name:' not in repr(s)

    def test_empty_name_printing(self):
        s = Series(index=date_range('20010101', '20020101'), name='test', dtype=object)
        assert 'Name: test' in repr(s)

    @pytest.mark.parametrize('args', [(), (0, -1)])
    def test_float_range(self, args):
        str(Series(np.random.randn(1000), index=np.arange(1000, *args)))

    def test_empty_object(self):
        str(Series(dtype=object))

    def test_string(self, string_series):
        str(string_series)
        str(string_series.astype(int))
        string_series[5:7] = np.NaN
        str(string_series)

    def test_object(self, object_series):
        str(object_series)

    def test_datetime(self, datetime_series):
        str(datetime_series)
        ots = datetime_series.astype('O')
        ots[::2] = None
        repr(ots)

    @pytest.mark.parametrize('name', ['', 1, 1.2, 'foo', 'αβγ', 'loooooooooooooooooooooooooooooooooooooooooooooooooooong', ('foo', 'bar', 'baz'), (1, 2), ('foo', 1, 2.3), ('α', 'β', 'γ'), ('α', 'bar')])
    def test_various_names(self, name, string_series):
        string_series.name = name
        repr(string_series)

    def test_tuple_name(self):
        biggie = Series(np.random.randn(1000), index=np.arange(1000), name=('foo', 'bar', 'baz'))
        repr(biggie)

    @pytest.mark.parametrize('arg', [100, 1001])
    def test_tidy_repr_name_0(self, arg):
        ser = Series(np.random.randn(arg), name=0)
        rep_str = repr(ser)
        assert 'Name: 0' in rep_str

    def test_newline(self):
        ser = Series(['a\n\r\tb'], name='a\n\r\td', index=['a\n\r\tf'])
        assert '\t' not in repr(ser)
        assert '\r' not in repr(ser)
        assert 'a\n' not in repr(ser)

    @pytest.mark.parametrize('name, expected', [['foo', 'Series([], Name: foo, dtype: int64)'], [None, 'Series([], dtype: int64)']])
    def test_empty_int64(self, name, expected):
        s = Series([], dtype=np.int64, name=name)
        assert repr(s) == expected

    def test_tidy_repr(self):
        a = Series(['א'] * 1000)
        a.name = 'title1'
        repr(a)

    def test_repr_bool_fails(self, capsys):
        s = Series([DataFrame(np.random.randn(2, 2)) for i in range(5)])
        repr(s)
        captured = capsys.readouterr()
        assert captured.err == ''

    def test_repr_name_iterable_indexable(self):
        s = Series([1, 2, 3], name=np.int64(3))
        repr(s)
        s.name = ('א',) * 2
        repr(s)

    def test_repr_should_return_str(self):
        data = [8, 5, 3, 5]
        index1 = ['σ', 'τ', 'υ', 'φ']
        df = Series(data, index=index1)
        assert type(df.__repr__() == str)

    def test_repr_max_rows(self):
        with option_context('display.max_rows', None):
            str(Series(range(1001)))

    def test_unicode_string_with_unicode(self):
        df = Series(['א'], name='ב')
        str(df)

    def test_str_to_bytes_raises(self):
        df = Series(['abc'], name='abc')
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(df)

    def test_timeseries_repr_object_dtype(self):
        index = Index([datetime(2000, 1, 1) + timedelta(i) for i in range(1000)], dtype=object)
        ts = Series(np.random.randn(len(index)), index)
        repr(ts)
        ts = tm.makeTimeSeries(1000)
        assert repr(ts).splitlines()[-1].startswith('Freq:')
        ts2 = ts.iloc[np.random.randint(0, len(ts) - 1, 400)]
        repr(ts2).splitlines()[-1]

    def test_latex_repr(self):
        pytest.importorskip('jinja2')
        result = '\\begin{tabular}{ll}\n\\toprule\n & 0 \\\\\n\\midrule\n0 & $\\alpha$ \\\\\n1 & b \\\\\n2 & c \\\\\n\\bottomrule\n\\end{tabular}\n'
        with option_context('styler.format.escape', None, 'styler.render.repr', 'latex'):
            s = Series(['$\\alpha$', 'b', 'c'])
            assert result == s._repr_latex_()
        assert s._repr_latex_() is None

    def test_index_repr_in_frame_with_nan(self):
        i = Index([1, np.nan])
        s = Series([1, 2], index=i)
        exp = '1.0    1\nNaN    2\ndtype: int64'
        assert repr(s) == exp

    def test_format_pre_1900_dates(self):
        rng = date_range('1/1/1850', '1/1/1950', freq='A-DEC')
        rng.format()
        ts = Series(1, index=rng)
        repr(ts)

    def test_series_repr_nat(self):
        series = Series([0, 1000, 2000, pd.NaT._value], dtype='M8[ns]')
        result = repr(series)
        expected = '0   1970-01-01 00:00:00.000000\n1   1970-01-01 00:00:00.000001\n2   1970-01-01 00:00:00.000002\n3                          NaT\ndtype: datetime64[ns]'
        assert result == expected

    def test_float_repr(self):
        ser = Series([1.0]).astype(object)
        expected = '0    1.0\ndtype: object'
        assert repr(ser) == expected

    def test_different_null_objects(self):
        ser = Series([1, 2, 3, 4], [True, None, np.nan, pd.NaT])
        result = repr(ser)
        expected = 'True    1\nNone    2\nNaN     3\nNaT     4\ndtype: int64'
        assert result == expected