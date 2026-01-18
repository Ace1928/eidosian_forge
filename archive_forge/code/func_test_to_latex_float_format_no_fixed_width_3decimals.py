import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_float_format_no_fixed_width_3decimals(self):
    df = DataFrame({'x': [0.19999]})
    result = df.to_latex(float_format='%.3f')
    expected = _dedent('\n            \\begin{tabular}{lr}\n            \\toprule\n             & x \\\\\n            \\midrule\n            0 & 0.200 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected