import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_empty_longtable(self):
    df = DataFrame()
    result = df.to_latex(longtable=True)
    expected = _dedent('\n            \\begin{longtable}{l}\n            \\toprule\n            \\midrule\n            \\endfirsthead\n            \\toprule\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{0}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            \\end{longtable}\n            ')
    assert result == expected