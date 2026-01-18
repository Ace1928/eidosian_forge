import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_series(self):
    s = Series(['a', 'b', 'c'])
    result = s.to_latex()
    expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & a \\\\\n            1 & b \\\\\n            2 & c \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected