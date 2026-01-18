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
def test_setitem_period_d_dtype(self):
    rng = period_range('2016-01-01', periods=9, freq='D', name='A')
    result = DataFrame(rng)
    expected = DataFrame({'A': ['NaT', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT']}, dtype='period[D]')
    result.iloc[:] = rng._na_value
    tm.assert_frame_equal(result, expected)