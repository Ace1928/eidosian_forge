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
@pytest.mark.parametrize('indexer', [tm.setitem, tm.loc])
def test_setitem_boolean_mask_aligning(self, indexer):
    df = DataFrame({'a': [1, 4, 2, 3], 'b': [5, 6, 7, 8]})
    expected = df.copy()
    mask = df['a'] >= 3
    indexer(df)[mask] = indexer(df)[mask].sort_values('a')
    tm.assert_frame_equal(df, expected)