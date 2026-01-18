import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_numexpr_with_min_and_max_columns(self):
    df = DataFrame({'min': [1, 2, 3], 'max': [4, 5, 6]})
    regex_to_match = 'Variables in expression \\"\\(min\\) == \\(1\\)\\" overlap with builtins: \\(\'min\'\\)'
    with pytest.raises(NumExprClobberingError, match=regex_to_match):
        df.query('min == 1')
    regex_to_match = 'Variables in expression \\"\\(max\\) == \\(1\\)\\" overlap with builtins: \\(\'max\'\\)'
    with pytest.raises(NumExprClobberingError, match=regex_to_match):
        df.query('max == 1')