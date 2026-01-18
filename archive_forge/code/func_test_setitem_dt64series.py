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
def test_setitem_dt64series(self, idx, expected):
    df = DataFrame(np.random.default_rng(2).standard_normal((2, 1)), columns=['A'])
    df['B'] = idx
    df['B'] = idx.to_series(index=[0, 1]).dt.tz_convert(None)
    result = df['B']
    comp = Series(idx.tz_convert('UTC').tz_localize(None), name='B')
    tm.assert_series_equal(result, comp)