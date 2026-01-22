from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.fixture
def arr_complex_nan(arr_complex, arr_nan_nanj):
    with np.errstate(invalid='ignore'):
        return np.vstack([arr_complex, arr_nan_nanj])