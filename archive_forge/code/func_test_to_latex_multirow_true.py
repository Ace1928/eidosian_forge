import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multirow_true(self, multicolumn_frame):
    result = multicolumn_frame.T.to_latex(multirow=True)
    expected = _dedent('\n            \\begin{tabular}{llrrrrr}\n            \\toprule\n             &  & 0 & 1 & 2 & 3 & 4 \\\\\n            \\midrule\n            \\multirow[t]{2}{*}{c1} & 0 & 0 & 1 & 2 & 3 & 4 \\\\\n             & 1 & 5 & 6 & 7 & 8 & 9 \\\\\n            \\cline{1-7}\n            \\multirow[t]{2}{*}{c2} & 0 & 0 & 1 & 2 & 3 & 4 \\\\\n             & 1 & 5 & 6 & 7 & 8 & 9 \\\\\n            \\cline{1-7}\n            c3 & 0 & 0 & 1 & 2 & 3 & 4 \\\\\n            \\cline{1-7}\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected