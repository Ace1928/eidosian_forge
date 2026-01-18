from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_tda_add_sub_index(self):
    tdi = TimedeltaIndex(['1 days', NaT, '2 days'])
    tda = tdi.array
    dti = pd.date_range('1999-12-31', periods=3, freq='D')
    result = tda + dti
    expected = tdi + dti
    tm.assert_index_equal(result, expected)
    result = tda + tdi
    expected = tdi + tdi
    tm.assert_index_equal(result, expected)
    result = tda - tdi
    expected = tdi - tdi
    tm.assert_index_equal(result, expected)