from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_stack_preserve_categorical_dtype_values(self, future_stack):
    cat = pd.Categorical(['a', 'a', 'b', 'c'])
    df = DataFrame({'A': cat, 'B': cat})
    result = df.stack(future_stack=future_stack)
    index = MultiIndex.from_product([[0, 1, 2, 3], ['A', 'B']])
    expected = Series(pd.Categorical(['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c']), index=index)
    tm.assert_series_equal(result, expected)