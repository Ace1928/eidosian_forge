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
def test_rdiv_zero(self):
    ser = Series([-1, 0, 1], name='first')
    expected = Series([0.0, np.nan, 0.0], name='first')
    result = 0 / ser
    tm.assert_series_equal(result, expected)