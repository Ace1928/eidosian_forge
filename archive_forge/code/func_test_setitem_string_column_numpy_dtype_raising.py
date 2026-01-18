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
def test_setitem_string_column_numpy_dtype_raising(self):
    df = DataFrame([[1, 2], [3, 4]])
    df['0 - Name'] = [5, 6]
    expected = DataFrame([[1, 2, 5], [3, 4, 6]], columns=[0, 1, '0 - Name'])
    tm.assert_frame_equal(df, expected)