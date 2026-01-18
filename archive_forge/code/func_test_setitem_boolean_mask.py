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
@td.skip_array_manager_invalid_test
@pytest.mark.parametrize('mask_type', [lambda df: df > np.abs(df) / 2, lambda df: (df > np.abs(df) / 2).values], ids=['dataframe', 'array'])
def test_setitem_boolean_mask(self, mask_type, float_frame):
    df = float_frame.copy()
    mask = mask_type(df)
    result = df.copy()
    result[mask] = np.nan
    expected = df.values.copy()
    expected[np.array(mask)] = np.nan
    expected = DataFrame(expected, index=df.index, columns=df.columns)
    tm.assert_frame_equal(result, expected)