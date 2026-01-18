from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('dtype', ['f8', 'i8', 'u8'])
def test_setitem_bool_with_numeric_index(self, dtype):
    cols = Index([1, 2, 3], dtype=dtype)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=cols)
    df[False] = ['a', 'b', 'c']
    expected_cols = Index([1, 2, 3, False], dtype=object)
    if dtype == 'f8':
        expected_cols = Index([1.0, 2.0, 3.0, False], dtype=object)
    tm.assert_index_equal(df.columns, expected_cols)