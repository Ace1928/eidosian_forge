import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_no_header_without_index(self):
    df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
    result = df.to_latex(index=False, header=False)
    expected = _dedent('\n            \\begin{tabular}{rl}\n            \\toprule\n            \\midrule\n            1 & b1 \\\\\n            2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected