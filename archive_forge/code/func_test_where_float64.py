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
@pytest.mark.parametrize('fill_val, exp_dtype', [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)])
def test_where_float64(self, index_or_series, fill_val, exp_dtype, request):
    klass = index_or_series
    obj = klass([1.1, 2.2, 3.3, 4.4])
    assert obj.dtype == np.float64
    self._run_test(obj, fill_val, klass, exp_dtype)