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
def test_setitem_empty_columns(self):
    df = DataFrame(index=['A', 'B', 'C'])
    df['X'] = df.index
    df['X'] = ['x', 'y', 'z']
    exp = DataFrame(data={'X': ['x', 'y', 'z']}, index=['A', 'B', 'C'])
    tm.assert_frame_equal(df, exp)