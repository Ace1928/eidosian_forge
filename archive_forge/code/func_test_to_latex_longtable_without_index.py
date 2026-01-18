import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_longtable_without_index(self):
    df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
    result = df.to_latex(index=False, longtable=True)
    expected = _dedent('\n            \\begin{longtable}{rl}\n            \\toprule\n            a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\toprule\n            a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{2}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            1 & b1 \\\\\n            2 & b2 \\\\\n            \\end{longtable}\n            ')
    assert result == expected