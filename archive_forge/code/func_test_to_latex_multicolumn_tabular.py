import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multicolumn_tabular(self, multiindex_frame):
    df = multiindex_frame.T
    df.columns.names = ['a', 'b']
    result = df.to_latex(multirow=False)
    expected = _dedent('\n            \\begin{tabular}{lrrrrr}\n            \\toprule\n            a & \\multicolumn{2}{r}{c1} & \\multicolumn{2}{r}{c2} & c3 \\\\\n            b & 0 & 1 & 0 & 1 & 0 \\\\\n            \\midrule\n            0 & 0 & 4 & 0 & 4 & 0 \\\\\n            1 & 1 & 5 & 1 & 5 & 1 \\\\\n            2 & 2 & 6 & 2 & 6 & 2 \\\\\n            3 & 3 & 7 & 3 & 7 & 3 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected