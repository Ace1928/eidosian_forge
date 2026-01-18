import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multiindex_small_tabular(self):
    df = DataFrame({('x', 'y'): ['a']}).T
    result = df.to_latex(multirow=False)
    expected = _dedent('\n            \\begin{tabular}{lll}\n            \\toprule\n             &  & 0 \\\\\n            \\midrule\n            x & y & a \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected