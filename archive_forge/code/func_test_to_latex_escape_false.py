import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_escape_false(self, df_with_symbols):
    result = df_with_symbols.to_latex(escape=False)
    expected = _dedent('\n            \\begin{tabular}{lll}\n            \\toprule\n             & co$e^x$ & co^l1 \\\\\n            \\midrule\n            a & a & a \\\\\n            b & b & b \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected