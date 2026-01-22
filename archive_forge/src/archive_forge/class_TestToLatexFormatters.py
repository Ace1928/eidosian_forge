import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestToLatexFormatters:

    def test_to_latex_with_formatters(self):
        df = DataFrame({'datetime64': [datetime(2016, 1, 1), datetime(2016, 2, 5), datetime(2016, 3, 3)], 'float': [1.0, 2.0, 3.0], 'int': [1, 2, 3], 'object': [(1, 2), True, False]})
        formatters = {'datetime64': lambda x: x.strftime('%Y-%m'), 'float': lambda x: f'[{x: 4.1f}]', 'int': lambda x: f'0x{x:x}', 'object': lambda x: f'-{x!s}-', '__index__': lambda x: f'index: {x}'}
        result = df.to_latex(formatters=dict(formatters))
        expected = _dedent('\n            \\begin{tabular}{llrrl}\n            \\toprule\n             & datetime64 & float & int & object \\\\\n            \\midrule\n            index: 0 & 2016-01 & [ 1.0] & 0x1 & -(1, 2)- \\\\\n            index: 1 & 2016-02 & [ 2.0] & 0x2 & -True- \\\\\n            index: 2 & 2016-03 & [ 3.0] & 0x3 & -False- \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_float_format_no_fixed_width_3decimals(self):
        df = DataFrame({'x': [0.19999]})
        result = df.to_latex(float_format='%.3f')
        expected = _dedent('\n            \\begin{tabular}{lr}\n            \\toprule\n             & x \\\\\n            \\midrule\n            0 & 0.200 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_float_format_no_fixed_width_integer(self):
        df = DataFrame({'x': [100.0]})
        result = df.to_latex(float_format='%.0f')
        expected = _dedent('\n            \\begin{tabular}{lr}\n            \\toprule\n             & x \\\\\n            \\midrule\n            0 & 100 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    @pytest.mark.parametrize('na_rep', ['NaN', 'Ted'])
    def test_to_latex_na_rep_and_float_format(self, na_rep):
        df = DataFrame([['A', 1.2225], ['A', None]], columns=['Group', 'Data'])
        result = df.to_latex(na_rep=na_rep, float_format='{:.2f}'.format)
        expected = _dedent(f'\n            \\begin{{tabular}}{{llr}}\n            \\toprule\n             & Group & Data \\\\\n            \\midrule\n            0 & A & 1.22 \\\\\n            1 & A & {na_rep} \\\\\n            \\bottomrule\n            \\end{{tabular}}\n            ')
        assert result == expected