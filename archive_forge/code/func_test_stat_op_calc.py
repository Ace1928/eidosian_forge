from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_stat_op_calc(self, float_frame_with_na, mixed_float_frame):

    def count(s):
        return notna(s).sum()

    def nunique(s):
        return len(algorithms.unique1d(s.dropna()))

    def var(x):
        return np.var(x, ddof=1)

    def std(x):
        return np.std(x, ddof=1)

    def sem(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))
    assert_stat_op_calc('nunique', nunique, float_frame_with_na, has_skipna=False, check_dtype=False, check_dates=True)
    assert_stat_op_calc('sum', np.sum, mixed_float_frame.astype('float32'), check_dtype=False, rtol=0.001)
    assert_stat_op_calc('sum', np.sum, float_frame_with_na, skipna_alternative=np.nansum)
    assert_stat_op_calc('mean', np.mean, float_frame_with_na, check_dates=True)
    assert_stat_op_calc('product', np.prod, float_frame_with_na, skipna_alternative=np.nanprod)
    assert_stat_op_calc('var', var, float_frame_with_na)
    assert_stat_op_calc('std', std, float_frame_with_na)
    assert_stat_op_calc('sem', sem, float_frame_with_na)
    assert_stat_op_calc('count', count, float_frame_with_na, has_skipna=False, check_dtype=False, check_dates=True)