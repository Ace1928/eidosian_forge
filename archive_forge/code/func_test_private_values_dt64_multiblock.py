import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_private_values_dt64_multiblock(self):
    dta = date_range('2000', periods=8)._data
    df = DataFrame({'A': dta[:4]}, copy=False)
    df['B'] = dta[4:]
    assert len(df._mgr.arrays) == 2
    result = df._values
    expected = dta.reshape(2, 4).T
    tm.assert_equal(result, expected)