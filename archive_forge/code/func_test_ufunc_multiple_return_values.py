from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('holder', [Index, Series])
@pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
def test_ufunc_multiple_return_values(self, holder, dtype):
    obj = holder([1, 2, 3], dtype=dtype, name='x')
    box = Series if holder is Series else Index
    result = np.modf(obj)
    assert isinstance(result, tuple)
    exp1 = Index([0.0, 0.0, 0.0], dtype=np.float64, name='x')
    exp2 = Index([1.0, 2.0, 3.0], dtype=np.float64, name='x')
    tm.assert_equal(result[0], tm.box_expected(exp1, box))
    tm.assert_equal(result[1], tm.box_expected(exp2, box))