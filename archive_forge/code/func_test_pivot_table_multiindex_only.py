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
@pytest.mark.parametrize('cols', [(1, 2), ('a', 'b'), (1, 'b'), ('a', 1)])
def test_pivot_table_multiindex_only(self, cols):
    df2 = DataFrame({cols[0]: [1, 2, 3], cols[1]: [1, 2, 3], 'v': [4, 5, 6]})
    result = df2.pivot_table(values='v', columns=cols)
    expected = DataFrame([[4.0, 5.0, 6.0]], columns=MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3)], names=cols), index=Index(['v'], dtype=object))
    tm.assert_frame_equal(result, expected)