import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_non_str(self):
    df = DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'b']})
    msg = 'expr must be a string to be evaluated'
    with pytest.raises(ValueError, match=msg):
        df.query(lambda x: x.B == 'b')
    with pytest.raises(ValueError, match=msg):
        df.query(111)