import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
from pandas import (
def test_period_dtype_match():
    dtype = PeriodDtype(freq='D')
    assert find_common_type([dtype, dtype]) == 'period[D]'