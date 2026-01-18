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
@pytest.mark.parametrize('op', [operator.pow, ops.rpow])
def test_pow_float(self, op, numeric_idx, box_with_array):
    box = box_with_array
    idx = numeric_idx
    expected = Index(op(idx.values, 2.0))
    idx = tm.box_expected(idx, box)
    expected = tm.box_expected(expected, box)
    result = op(idx, 2.0)
    tm.assert_equal(result, expected)