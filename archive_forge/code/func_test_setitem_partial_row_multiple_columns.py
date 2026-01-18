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
def test_setitem_partial_row_multiple_columns():
    df = DataFrame({'A': [1, 2, 3], 'B': [4.0, 5, 6]})
    df.loc[df.index <= 1, ['F', 'G']] = (1, 'abc')
    expected = DataFrame({'A': [1, 2, 3], 'B': [4.0, 5, 6], 'F': [1.0, 1, float('nan')], 'G': ['abc', 'abc', float('nan')]})
    tm.assert_frame_equal(df, expected)