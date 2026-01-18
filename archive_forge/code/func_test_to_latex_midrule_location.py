import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_midrule_location(self):
    df = DataFrame({'a': [1, 2]})
    df.index.name = 'foo'
    result = df.to_latex(index_names=False)
    expected = _dedent('\n            \\begin{tabular}{lr}\n            \\toprule\n             & a \\\\\n            \\midrule\n            0 & 1 \\\\\n            1 & 2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected