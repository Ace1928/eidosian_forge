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
def test_pivot_integer_columns(self):
    d = date.min
    data = list(product(['foo', 'bar'], ['A', 'B', 'C'], ['x1', 'x2'], [d + timedelta(i) for i in range(20)], [1.0]))
    df = DataFrame(data)
    table = df.pivot_table(values=4, index=[0, 1, 3], columns=[2])
    df2 = df.rename(columns=str)
    table2 = df2.pivot_table(values='4', index=['0', '1', '3'], columns=['2'])
    tm.assert_frame_equal(table, table2, check_names=False)