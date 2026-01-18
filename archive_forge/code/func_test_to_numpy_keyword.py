from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_to_numpy_keyword():
    values = [decimal.Decimal('1.1111'), decimal.Decimal('2.2222')]
    expected = np.array([decimal.Decimal('1.11'), decimal.Decimal('2.22')], dtype='object')
    a = pd.array(values, dtype='decimal')
    result = a.to_numpy(decimals=2)
    tm.assert_numpy_array_equal(result, expected)
    result = pd.Series(a).to_numpy(decimals=2)
    tm.assert_numpy_array_equal(result, expected)