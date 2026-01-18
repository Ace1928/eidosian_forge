from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_table_with_margins_and_numeric_columns(self):
    df = DataFrame([['a', 'x', 1], ['a', 'y', 2], ['b', 'y', 3], ['b', 'z', 4]])
    df.columns = [10, 20, 30]
    result = df.pivot_table(index=10, columns=20, values=30, aggfunc='sum', fill_value=0, margins=True)
    expected = DataFrame([[1, 2, 0, 3], [0, 3, 4, 7], [1, 5, 4, 10]])
    expected.columns = ['x', 'y', 'z', 'All']
    expected.index = ['a', 'b', 'All']
    expected.columns.name = 20
    expected.index.name = 10
    tm.assert_frame_equal(result, expected)