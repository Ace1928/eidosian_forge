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
@pytest.mark.parametrize('dtype', [object, 'string'])
def test_pivot_integer_bug(self, dtype):
    df = DataFrame(data=[('A', '1', 'A1'), ('B', '2', 'B2')], dtype=dtype)
    result = df.pivot(index=1, columns=0, values=2)
    tm.assert_index_equal(result.columns, Index(['A', 'B'], name=0, dtype=dtype))