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
def test_setitem_other_callable(self):

    def inc(x):
        return x + 1
    df = DataFrame([[-1, 1], [1, -1]], dtype=object)
    df[df > 0] = inc
    expected = DataFrame([[-1, inc], [inc, -1]])
    tm.assert_frame_equal(df, expected)