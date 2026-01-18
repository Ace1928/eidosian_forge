import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_caption_shortcaption_and_label(self, df_short, caption_table, short_caption, label_table):
    result = df_short.to_latex(caption=(caption_table, short_caption), label=label_table)
    expected = _dedent('\n            \\begin{table}\n            \\caption[a table]{a table in a \\texttt{table/tabular} environment}\n            \\label{tab:table_tabular}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
    assert result == expected