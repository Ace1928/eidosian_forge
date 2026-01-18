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
@pytest.mark.parametrize('dtype', ['int32', 'int64', 'uint32', 'uint64', 'float32', 'float64'])
def test_setitem_dtype(self, dtype, float_frame):
    arr = np.random.default_rng(2).integers(1, 10, len(float_frame))
    float_frame[dtype] = np.array(arr, dtype=dtype)
    assert float_frame[dtype].dtype.name == dtype