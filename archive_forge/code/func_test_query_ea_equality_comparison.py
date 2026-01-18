import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('engine', ['python', 'numexpr'])
@pytest.mark.parametrize('dtype', ['int64', 'Int64', 'int64[pyarrow]'])
def test_query_ea_equality_comparison(self, dtype, engine):
    warning = RuntimeWarning if engine == 'numexpr' else None
    if engine == 'numexpr' and (not NUMEXPR_INSTALLED):
        pytest.skip('numexpr not installed')
    if dtype == 'int64[pyarrow]':
        pytest.importorskip('pyarrow')
    df = DataFrame({'A': Series([1, 1, 2], dtype='Int64'), 'B': Series([1, 2, 2], dtype=dtype)})
    with tm.assert_produces_warning(warning):
        result = df.query('A == B', engine=engine)
    expected = DataFrame({'A': Series([1, 2], dtype='Int64', index=[0, 2]), 'B': Series([1, 2], dtype=dtype, index=[0, 2])})
    tm.assert_frame_equal(result, expected)