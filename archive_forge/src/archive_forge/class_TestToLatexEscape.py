import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestToLatexEscape:

    @pytest.fixture
    def df_with_symbols(self):
        """Dataframe with special characters for testing chars escaping."""
        a = 'a'
        b = 'b'
        yield DataFrame({'co$e^x$': {a: 'a', b: 'b'}, 'co^l1': {a: 'a', b: 'b'}})

    def test_to_latex_escape_false(self, df_with_symbols):
        result = df_with_symbols.to_latex(escape=False)
        expected = _dedent('\n            \\begin{tabular}{lll}\n            \\toprule\n             & co$e^x$ & co^l1 \\\\\n            \\midrule\n            a & a & a \\\\\n            b & b & b \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_escape_default(self, df_with_symbols):
        default = df_with_symbols.to_latex()
        specified_true = df_with_symbols.to_latex(escape=True)
        assert default != specified_true

    def test_to_latex_special_escape(self):
        df = DataFrame(['a\\b\\c', '^a^b^c', '~a~b~c'])
        result = df.to_latex(escape=True)
        expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & a\\textbackslash b\\textbackslash c \\\\\n            1 & \\textasciicircum a\\textasciicircum b\\textasciicircum c \\\\\n            2 & \\textasciitilde a\\textasciitilde b\\textasciitilde c \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_escape_special_chars(self):
        special_characters = ['&', '%', '$', '#', '_', '{', '}', '~', '^', '\\']
        df = DataFrame(data=special_characters)
        result = df.to_latex(escape=True)
        expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & \\& \\\\\n            1 & \\% \\\\\n            2 & \\$ \\\\\n            3 & \\# \\\\\n            4 & \\_ \\\\\n            5 & \\{ \\\\\n            6 & \\} \\\\\n            7 & \\textasciitilde  \\\\\n            8 & \\textasciicircum  \\\\\n            9 & \\textbackslash  \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_specified_header_special_chars_without_escape(self):
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(header=['$A$', '$B$'], escape=False)
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & $A$ & $B$ \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected