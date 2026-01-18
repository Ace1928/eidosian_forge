from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('val,exp_dtype', [(5, np.float64), (5.1, np.float64), ('x', object)])
def test_setitem_index_float64(self, val, exp_dtype, request):
    obj = pd.Series([1, 2, 3, 4], index=[1.1, 2.1, 3.1, 4.1])
    assert obj.index.dtype == np.float64
    exp_index = pd.Index([1.1, 2.1, 3.1, 4.1, val])
    self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)