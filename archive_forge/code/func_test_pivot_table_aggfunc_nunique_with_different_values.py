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
def test_pivot_table_aggfunc_nunique_with_different_values(self):
    test = DataFrame({'a': range(10), 'b': range(10), 'c': range(10), 'd': range(10)})
    columnval = MultiIndex.from_arrays([['nunique' for i in range(10)], ['c' for i in range(10)], range(10)], names=(None, None, 'b'))
    nparr = np.full((10, 10), np.nan)
    np.fill_diagonal(nparr, 1.0)
    expected = DataFrame(nparr, index=Index(range(10), name='a'), columns=columnval)
    result = test.pivot_table(index=['a'], columns=['b'], values=['c'], aggfunc=['nunique'])
    tm.assert_frame_equal(result, expected)