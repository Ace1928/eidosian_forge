from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('op, equiv_op, fv', pairings)
def test_operators_combine(self, op, equiv_op, fv):

    def _check_fill(meth, op, a, b, fill_value=0):
        exp_index = a.index.union(b.index)
        a = a.reindex(exp_index)
        b = b.reindex(exp_index)
        amask = isna(a)
        bmask = isna(b)
        exp_values = []
        for i in range(len(exp_index)):
            with np.errstate(all='ignore'):
                if amask[i]:
                    if bmask[i]:
                        exp_values.append(np.nan)
                        continue
                    exp_values.append(op(fill_value, b[i]))
                elif bmask[i]:
                    if amask[i]:
                        exp_values.append(np.nan)
                        continue
                    exp_values.append(op(a[i], fill_value))
                else:
                    exp_values.append(op(a[i], b[i]))
        result = meth(a, b, fill_value=fill_value)
        expected = Series(exp_values, exp_index)
        tm.assert_series_equal(result, expected)
    a = Series([np.nan, 1.0, 2.0, 3.0, np.nan], index=np.arange(5))
    b = Series([np.nan, 1, np.nan, 3, np.nan, 4.0], index=np.arange(6))
    result = op(a, b)
    exp = equiv_op(a, b)
    tm.assert_series_equal(result, exp)
    _check_fill(op, equiv_op, a, b, fill_value=fv)
    op(a, b, axis=0)