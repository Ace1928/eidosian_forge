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
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason='None is cast to NaN')
def test_pivot_index_is_none(self):
    df = DataFrame({None: [1], 'b': 2, 'c': 3})
    result = df.pivot(columns='b', index=None)
    expected = DataFrame({('c', 2): 3}, index=[1])
    expected.columns.names = [None, 'b']
    tm.assert_frame_equal(result, expected)
    result = df.pivot(columns='b', index=None, values='c')
    expected = DataFrame(3, index=[1], columns=Index([2], name='b'))
    tm.assert_frame_equal(result, expected)