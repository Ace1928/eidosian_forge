from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_take_na_value_other_decimal():
    arr = DecimalArray([decimal.Decimal('1.0'), decimal.Decimal('2.0')])
    result = arr.take([0, -1], allow_fill=True, fill_value=decimal.Decimal('-1.0'))
    expected = DecimalArray([decimal.Decimal('1.0'), decimal.Decimal('-1.0')])
    tm.assert_extension_array_equal(result, expected)