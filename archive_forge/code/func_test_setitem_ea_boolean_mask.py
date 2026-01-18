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
def test_setitem_ea_boolean_mask(self):
    df = DataFrame([[-1, 2], [3, -4]])
    expected = DataFrame([[0, 2], [3, 0]])
    boolean_indexer = DataFrame({0: Series([True, False], dtype='boolean'), 1: Series([pd.NA, True], dtype='boolean')})
    df[boolean_indexer] = 0
    tm.assert_frame_equal(df, expected)