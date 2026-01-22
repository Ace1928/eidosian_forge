from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
class TestDataFrameToStringFormatters:

    def test_to_string_masked_ea_with_formatter(self):
        df = DataFrame({'a': Series([0.123456789, 1.123456789], dtype='Float64'), 'b': Series([1, 2], dtype='Int64')})
        result = df.to_string(formatters=['{:.2f}'.format, '{:.2f}'.format])
        expected = dedent('                  a     b\n            0  0.12  1.00\n            1  1.12  2.00')
        assert result == expected

    def test_to_string_with_formatters(self):
        df = DataFrame({'int': [1, 2, 3], 'float': [1.0, 2.0, 3.0], 'object': [(1, 2), True, False]}, columns=['int', 'float', 'object'])
        formatters = [('int', lambda x: f'0x{x:x}'), ('float', lambda x: f'[{x: 4.1f}]'), ('object', lambda x: f'-{x!s}-')]
        result = df.to_string(formatters=dict(formatters))
        result2 = df.to_string(formatters=list(zip(*formatters))[1])
        assert result == '  int  float    object\n0 0x1 [ 1.0]  -(1, 2)-\n1 0x2 [ 2.0]    -True-\n2 0x3 [ 3.0]   -False-'
        assert result == result2

    def test_to_string_with_datetime64_monthformatter(self):
        months = [datetime(2016, 1, 1), datetime(2016, 2, 2)]
        x = DataFrame({'months': months})

        def format_func(x):
            return x.strftime('%Y-%m')
        result = x.to_string(formatters={'months': format_func})
        expected = dedent('            months\n            0 2016-01\n            1 2016-02')
        assert result.strip() == expected

    def test_to_string_with_datetime64_hourformatter(self):
        x = DataFrame({'hod': to_datetime(['10:10:10.100', '12:12:12.120'], format='%H:%M:%S.%f')})

        def format_func(x):
            return x.strftime('%H:%M')
        result = x.to_string(formatters={'hod': format_func})
        expected = dedent('            hod\n            0 10:10\n            1 12:12')
        assert result.strip() == expected

    def test_to_string_with_formatters_unicode(self):
        df = DataFrame({'c/σ': [1, 2, 3]})
        result = df.to_string(formatters={'c/σ': str})
        expected = dedent('              c/σ\n            0   1\n            1   2\n            2   3')
        assert result == expected

        def test_to_string_index_formatter(self):
            df = DataFrame([range(5), range(5, 10), range(10, 15)])
            rs = df.to_string(formatters={'__index__': lambda x: 'abc'[x]})
            xp = dedent('                0   1   2   3   4\n            a   0   1   2   3   4\n            b   5   6   7   8   9\n            c  10  11  12  13  14            ')
            assert rs == xp

    def test_no_extra_space(self):
        col1 = 'TEST'
        col2 = 'PANDAS'
        col3 = 'to_string'
        expected = f'{col1:<6s} {col2:<7s} {col3:<10s}'
        df = DataFrame([{'col1': 'TEST', 'col2': 'PANDAS', 'col3': 'to_string'}])
        d = {'col1': '{:<6s}'.format, 'col2': '{:<7s}'.format, 'col3': '{:<10s}'.format}
        result = df.to_string(index=False, header=False, formatters=d)
        assert result == expected