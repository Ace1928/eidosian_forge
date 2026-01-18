import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_order_aggregate_multiple_funcs():
    df = DataFrame({'A': [1, 1, 2, 2], 'B': [1, 2, 3, 4]})
    res = df.groupby('A').agg(['sum', 'max', 'mean', 'ohlc', 'min'])
    result = res.columns.levels[1]
    expected = Index(['sum', 'max', 'mean', 'ohlc', 'min'])
    tm.assert_index_equal(result, expected)