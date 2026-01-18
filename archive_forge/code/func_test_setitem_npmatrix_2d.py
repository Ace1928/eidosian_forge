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
@td.skip_array_manager_not_yet_implemented
def test_setitem_npmatrix_2d(self):
    expected = DataFrame({'np-array': np.ones(10), 'np-matrix': np.ones(10)}, index=np.arange(10))
    a = np.ones((10, 1))
    df = DataFrame(index=np.arange(10))
    df['np-array'] = a
    with tm.assert_produces_warning(PendingDeprecationWarning):
        df['np-matrix'] = np.matrix(a)
    tm.assert_frame_equal(df, expected)