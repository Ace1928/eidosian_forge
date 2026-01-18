from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('two', [2, 2.0, np.array(2), np.array(2.0)])
def test_td64arr_floordiv_numeric_scalar(self, box_with_array, two):
    tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
    expected = Series(['29.5D', '29.5D', 'NaT'], dtype='timedelta64[ns]')
    tdser = tm.box_expected(tdser, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = tdser // two
    tm.assert_equal(result, expected)
    with pytest.raises(TypeError, match='Cannot divide'):
        two // tdser