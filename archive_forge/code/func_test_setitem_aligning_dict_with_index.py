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
@pytest.mark.parametrize('vals', [{}, {'d': 'a'}])
def test_setitem_aligning_dict_with_index(self, vals):
    df = DataFrame({'a': [1, 2], 'b': [3, 4], **vals})
    df.loc[:, 'a'] = {1: 100, 0: 200}
    df.loc[:, 'c'] = {0: 5, 1: 6}
    df.loc[:, 'e'] = {1: 5}
    expected = DataFrame({'a': [200, 100], 'b': [3, 4], **vals, 'c': [5, 6], 'e': [np.nan, 5]})
    tm.assert_frame_equal(df, expected)