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
def test_setitem_benchmark(self):
    N = 10
    K = 5
    df = DataFrame(index=range(N))
    new_col = np.random.default_rng(2).standard_normal(N)
    for i in range(K):
        df[i] = new_col
    expected = DataFrame(np.repeat(new_col, K).reshape(N, K), index=range(N))
    tm.assert_frame_equal(df, expected)