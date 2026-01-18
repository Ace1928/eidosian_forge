import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_no_bold_rows(self):
    df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
    result = df.to_latex(bold_rows=False)
    expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected