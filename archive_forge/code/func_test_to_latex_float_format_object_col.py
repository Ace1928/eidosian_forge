import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_float_format_object_col(self):
    ser = Series([1000.0, 'test'])
    result = ser.to_latex(float_format='{:,.0f}'.format)
    expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & 1,000 \\\\\n            1 & test \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected