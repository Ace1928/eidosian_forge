import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_empty_string(self):
    df = DataFrame({'A': [1, 2, 3]})
    msg = 'expr cannot be an empty string'
    with pytest.raises(ValueError, match=msg):
        df.query('')