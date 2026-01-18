from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
@pytest.mark.parametrize('n', [1, 10, 32, 100, 1000])
def test_sum_of_booleans(n):
    df = DataFrame({'groupby_col': 1, 'bool': [True] * n})
    df['bool'] = df['bool'].eq(True)
    result = df.groupby('groupby_col').sum()
    expected = DataFrame({'bool': [n]}, index=Index([1], name='groupby_col'))
    tm.assert_frame_equal(result, expected)