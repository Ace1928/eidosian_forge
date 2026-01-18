import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multindex_header(self):
    df = DataFrame({'a': [0], 'b': [1], 'c': [2], 'd': [3]})
    df = df.set_index(['a', 'b'])
    observed = df.to_latex(header=['r1', 'r2'], multirow=False)
    expected = _dedent('\n            \\begin{tabular}{llrr}\n            \\toprule\n             &  & r1 & r2 \\\\\n            a & b &  &  \\\\\n            \\midrule\n            0 & 1 & 2 & 3 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert observed == expected