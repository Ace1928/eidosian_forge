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
@pytest.mark.parametrize('dtype', ['float', 'int64'])
@pytest.mark.parametrize('kwargs', [{}, {'index': [1]}, {'columns': ['A']}])
def test_setitem_empty_frame_with_boolean(self, dtype, kwargs):
    kwargs['dtype'] = dtype
    df = DataFrame(**kwargs)
    df2 = df.copy()
    df[df > df2] = 47
    tm.assert_frame_equal(df, df2)