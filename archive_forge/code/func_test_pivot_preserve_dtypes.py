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
@pytest.mark.parametrize('columns,values', [('bool1', ['float1', 'float2']), ('bool1', ['float1', 'float2', 'bool1']), ('bool2', ['float1', 'float2', 'bool1'])])
def test_pivot_preserve_dtypes(self, columns, values):
    v = np.arange(5, dtype=np.float64)
    df = DataFrame({'float1': v, 'float2': v + 2.0, 'bool1': v <= 2, 'bool2': v <= 3})
    df_res = df.reset_index().pivot_table(index='index', columns=columns, values=values)
    result = dict(df_res.dtypes)
    expected = {col: np.dtype('float64') for col in df_res}
    assert result == expected