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
@pytest.mark.parametrize('target_add,input_value,expected_value', [('!', ['hello', 'world'], ['hello!', 'world!']), ('m', ['hello', 'world'], ['hellom', 'worldm'])])
def test_string_addition(self, target_add, input_value, expected_value):
    a = Series(input_value)
    result = a + target_add
    expected = Series(expected_value)
    tm.assert_series_equal(result, expected)