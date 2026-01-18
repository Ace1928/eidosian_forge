import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_multicolumn_default(self, multicolumn_frame):
    result = multicolumn_frame.to_latex()
    expected = _dedent('\n            \\begin{tabular}{lrrrrr}\n            \\toprule\n             & \\multicolumn{2}{r}{c1} & \\multicolumn{2}{r}{c2} & c3 \\\\\n             & 0 & 1 & 0 & 1 & 0 \\\\\n            \\midrule\n            0 & 0 & 5 & 0 & 5 & 0 \\\\\n            1 & 1 & 6 & 1 & 6 & 1 \\\\\n            2 & 2 & 7 & 2 & 7 & 2 \\\\\n            3 & 3 & 8 & 3 & 8 & 3 \\\\\n            4 & 4 & 9 & 4 & 9 & 4 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected