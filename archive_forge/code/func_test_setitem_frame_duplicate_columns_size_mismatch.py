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
def test_setitem_frame_duplicate_columns_size_mismatch(self):
    cols = ['A', 'B', 'C'] * 2
    df = DataFrame(index=range(3), columns=cols)
    with pytest.raises(ValueError, match='Columns must be same length as key'):
        df[['A']] = (0, 3, 5)
    df2 = df.iloc[:, :3]
    with pytest.raises(ValueError, match='Columns must be same length as key'):
        df2[['A']] = (0, 3, 5)