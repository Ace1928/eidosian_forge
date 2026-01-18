import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('dtype', ['int64', 'Int64', 'int64[pyarrow]'])
def test_query_ea_dtypes(self, dtype):
    if dtype == 'int64[pyarrow]':
        pytest.importorskip('pyarrow')
    df = DataFrame({'a': Series([1, 2], dtype=dtype)})
    ref = {2}
    warning = RuntimeWarning if dtype == 'Int64' and NUMEXPR_INSTALLED else None
    with tm.assert_produces_warning(warning):
        result = df.query('a in @ref')
    expected = DataFrame({'a': Series([2], dtype=dtype, index=[1])})
    tm.assert_frame_equal(result, expected)