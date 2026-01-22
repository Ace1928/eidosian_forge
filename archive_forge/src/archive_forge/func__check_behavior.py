from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def _check_behavior(self, arr, expected):
    result = libmissing.isnaobj(arr)
    tm.assert_numpy_array_equal(result, expected)
    result = libmissing.isnaobj(arr, inf_as_na=True)
    tm.assert_numpy_array_equal(result, expected)
    arr = np.atleast_2d(arr)
    expected = np.atleast_2d(expected)
    result = libmissing.isnaobj(arr)
    tm.assert_numpy_array_equal(result, expected)
    result = libmissing.isnaobj(arr, inf_as_na=True)
    tm.assert_numpy_array_equal(result, expected)
    arr = arr.copy(order='F')
    result = libmissing.isnaobj(arr)
    tm.assert_numpy_array_equal(result, expected)
    result = libmissing.isnaobj(arr, inf_as_na=True)
    tm.assert_numpy_array_equal(result, expected)