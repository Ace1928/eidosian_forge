from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def _skew_kurt_wrap(self, values, axis=None, func=None):
    if not isinstance(values.dtype.type, np.floating):
        values = values.astype('f8')
    result = func(values, axis=axis, bias=False)
    if isinstance(result, np.ndarray):
        result[np.max(values, axis=axis) == np.min(values, axis=axis)] = 0
        return result
    elif np.max(values) == np.min(values):
        return 0.0
    return result