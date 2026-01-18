from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_sum_inf(self):
    s = Series(np.random.default_rng(2).standard_normal(10))
    s2 = s.copy()
    s[5:8] = np.inf
    s2[5:8] = np.nan
    assert np.isinf(s.sum())
    arr = np.random.default_rng(2).standard_normal((100, 100)).astype('f4')
    arr[:, 2] = np.inf
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.use_inf_as_na', True):
            tm.assert_almost_equal(s.sum(), s2.sum())
    res = nanops.nansum(arr, axis=1)
    assert np.isinf(res).all()