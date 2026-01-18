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
def test_setitem_always_copy(self, float_frame):
    assert 'E' not in float_frame.columns
    s = float_frame['A'].copy()
    float_frame['E'] = s
    float_frame.iloc[5:10, float_frame.columns.get_loc('E')] = np.nan
    assert notna(s[5:10]).all()