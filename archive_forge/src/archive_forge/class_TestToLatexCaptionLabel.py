import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestToLatexCaptionLabel:

    @pytest.fixture
    def caption_table(self):
        """Caption for table/tabular LaTeX environment."""
        return 'a table in a \\texttt{table/tabular} environment'

    @pytest.fixture
    def short_caption(self):
        """Short caption for testing \\caption[short_caption]{full_caption}."""
        return 'a table'

    @pytest.fixture
    def label_table(self):
        """Label for table/tabular LaTeX environment."""
        return 'tab:table_tabular'

    @pytest.fixture
    def caption_longtable(self):
        """Caption for longtable LaTeX environment."""
        return 'a table in a \\texttt{longtable} environment'

    @pytest.fixture
    def label_longtable(self):
        """Label for longtable LaTeX environment."""
        return 'tab:longtable'

    def test_to_latex_caption_only(self, df_short, caption_table):
        result = df_short.to_latex(caption=caption_table)
        expected = _dedent('\n            \\begin{table}\n            \\caption{a table in a \\texttt{table/tabular} environment}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_label_only(self, df_short, label_table):
        result = df_short.to_latex(label=label_table)
        expected = _dedent('\n            \\begin{table}\n            \\label{tab:table_tabular}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_caption_and_label(self, df_short, caption_table, label_table):
        result = df_short.to_latex(caption=caption_table, label=label_table)
        expected = _dedent('\n            \\begin{table}\n            \\caption{a table in a \\texttt{table/tabular} environment}\n            \\label{tab:table_tabular}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_caption_and_shortcaption(self, df_short, caption_table, short_caption):
        result = df_short.to_latex(caption=(caption_table, short_caption))
        expected = _dedent('\n            \\begin{table}\n            \\caption[a table]{a table in a \\texttt{table/tabular} environment}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_caption_and_shortcaption_list_is_ok(self, df_short):
        caption = ('Long-long-caption', 'Short')
        result_tuple = df_short.to_latex(caption=caption)
        result_list = df_short.to_latex(caption=list(caption))
        assert result_tuple == result_list

    def test_to_latex_caption_shortcaption_and_label(self, df_short, caption_table, short_caption, label_table):
        result = df_short.to_latex(caption=(caption_table, short_caption), label=label_table)
        expected = _dedent('\n            \\begin{table}\n            \\caption[a table]{a table in a \\texttt{table/tabular} environment}\n            \\label{tab:table_tabular}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    @pytest.mark.parametrize('bad_caption', [('full_caption', 'short_caption', 'extra_string'), ('full_caption', 'short_caption', 1), ('full_caption', 'short_caption', None), ('full_caption',), (None,)])
    def test_to_latex_bad_caption_raises(self, bad_caption):
        df = DataFrame({'a': [1]})
        msg = '`caption` must be either a string or 2-tuple of strings'
        with pytest.raises(ValueError, match=msg):
            df.to_latex(caption=bad_caption)

    def test_to_latex_two_chars_caption(self, df_short):
        result = df_short.to_latex(caption='xy')
        expected = _dedent('\n            \\begin{table}\n            \\caption{xy}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_longtable_caption_only(self, df_short, caption_longtable):
        result = df_short.to_latex(longtable=True, caption=caption_longtable)
        expected = _dedent('\n            \\begin{longtable}{lrl}\n            \\caption{a table in a \\texttt{longtable} environment} \\\\\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\caption[]{a table in a \\texttt{longtable} environment} \\\\\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{3}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected

    def test_to_latex_longtable_label_only(self, df_short, label_longtable):
        result = df_short.to_latex(longtable=True, label=label_longtable)
        expected = _dedent('\n            \\begin{longtable}{lrl}\n            \\label{tab:longtable} \\\\\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{3}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected

    def test_to_latex_longtable_caption_and_label(self, df_short, caption_longtable, label_longtable):
        result = df_short.to_latex(longtable=True, caption=caption_longtable, label=label_longtable)
        expected = _dedent('\n        \\begin{longtable}{lrl}\n        \\caption{a table in a \\texttt{longtable} environment} \\label{tab:longtable} \\\\\n        \\toprule\n         & a & b \\\\\n        \\midrule\n        \\endfirsthead\n        \\caption[]{a table in a \\texttt{longtable} environment} \\\\\n        \\toprule\n         & a & b \\\\\n        \\midrule\n        \\endhead\n        \\midrule\n        \\multicolumn{3}{r}{Continued on next page} \\\\\n        \\midrule\n        \\endfoot\n        \\bottomrule\n        \\endlastfoot\n        0 & 1 & b1 \\\\\n        1 & 2 & b2 \\\\\n        \\end{longtable}\n        ')
        assert result == expected

    def test_to_latex_longtable_caption_shortcaption_and_label(self, df_short, caption_longtable, short_caption, label_longtable):
        result = df_short.to_latex(longtable=True, caption=(caption_longtable, short_caption), label=label_longtable)
        expected = _dedent('\n\\begin{longtable}{lrl}\n\\caption[a table]{a table in a \\texttt{longtable} environment} \\label{tab:longtable} \\\\\n\\toprule\n & a & b \\\\\n\\midrule\n\\endfirsthead\n\\caption[]{a table in a \\texttt{longtable} environment} \\\\\n\\toprule\n & a & b \\\\\n\\midrule\n\\endhead\n\\midrule\n\\multicolumn{3}{r}{Continued on next page} \\\\\n\\midrule\n\\endfoot\n\\bottomrule\n\\endlastfoot\n0 & 1 & b1 \\\\\n1 & 2 & b2 \\\\\n\\end{longtable}\n')
        assert result == expected