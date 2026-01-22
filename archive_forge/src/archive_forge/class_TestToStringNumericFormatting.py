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
class TestToStringNumericFormatting:

    def test_to_string_float_format_no_fixed_width(self):
        df = DataFrame({'x': [0.19999]})
        expected = '      x\n0 0.200'
        assert df.to_string(float_format='%.3f') == expected
        df = DataFrame({'x': [100.0]})
        expected = '    x\n0 100'
        assert df.to_string(float_format='%.0f') == expected

    def test_to_string_small_float_values(self):
        df = DataFrame({'a': [1.5, 1e-17, -5.5e-07]})
        result = df.to_string()
        if _three_digit_exp():
            expected = '               a\n0  1.500000e+000\n1  1.000000e-017\n2 -5.500000e-007'
        else:
            expected = '              a\n0  1.500000e+00\n1  1.000000e-17\n2 -5.500000e-07'
        assert result == expected
        df = df * 0
        result = df.to_string()
        expected = '   0\n0  0\n1  0\n2 -0'

    def test_to_string_complex_float_formatting(self):
        with option_context('display.precision', 5):
            df = DataFrame({'x': [0.4467846931321966 + 0.0715185102060818j, 0.2739442392974528 + 0.23515228785438969j, 0.26974928742135185 + 0.3250604054898979j, -1j]})
            result = df.to_string()
            expected = '                  x\n0  0.44678+0.07152j\n1  0.27394+0.23515j\n2  0.26975+0.32506j\n3 -0.00000-1.00000j'
            assert result == expected

    def test_to_string_format_inf(self):
        df = DataFrame({'A': [-np.inf, np.inf, -1, -2.1234, 3, 4], 'B': [-np.inf, np.inf, 'foo', 'foooo', 'fooooo', 'bar']})
        result = df.to_string()
        expected = '        A       B\n0    -inf    -inf\n1     inf     inf\n2 -1.0000     foo\n3 -2.1234   foooo\n4  3.0000  fooooo\n5  4.0000     bar'
        assert result == expected
        df = DataFrame({'A': [-np.inf, np.inf, -1.0, -2.0, 3.0, 4.0], 'B': [-np.inf, np.inf, 'foo', 'foooo', 'fooooo', 'bar']})
        result = df.to_string()
        expected = '     A       B\n0 -inf    -inf\n1  inf     inf\n2 -1.0     foo\n3 -2.0   foooo\n4  3.0  fooooo\n5  4.0     bar'
        assert result == expected

    def test_to_string_int_formatting(self):
        df = DataFrame({'x': [-15, 20, 25, -35]})
        assert issubclass(df['x'].dtype.type, np.integer)
        output = df.to_string()
        expected = '    x\n0 -15\n1  20\n2  25\n3 -35'
        assert output == expected

    def test_to_string_float_formatting(self):
        with option_context('display.precision', 5, 'display.notebook_repr_html', False):
            df = DataFrame({'x': [0, 0.25, 3456.0, 1.2e+46, 1640000.0, 170000000.0, 1.253456, np.pi, -1000000.0]})
            df_s = df.to_string()
            if _three_digit_exp():
                expected = '              x\n0  0.00000e+000\n1  2.50000e-001\n2  3.45600e+003\n3  1.20000e+046\n4  1.64000e+006\n5  1.70000e+008\n6  1.25346e+000\n7  3.14159e+000\n8 -1.00000e+006'
            else:
                expected = '             x\n0  0.00000e+00\n1  2.50000e-01\n2  3.45600e+03\n3  1.20000e+46\n4  1.64000e+06\n5  1.70000e+08\n6  1.25346e+00\n7  3.14159e+00\n8 -1.00000e+06'
            assert df_s == expected
            df = DataFrame({'x': [3234, 0.253]})
            df_s = df.to_string()
            expected = '          x\n0  3234.000\n1     0.253'
            assert df_s == expected
        assert get_option('display.precision') == 6
        df = DataFrame({'x': [1000000000.0, 0.2512]})
        df_s = df.to_string()
        if _three_digit_exp():
            expected = '               x\n0  1.000000e+009\n1  2.512000e-001'
        else:
            expected = '              x\n0  1.000000e+09\n1  2.512000e-01'
        assert df_s == expected