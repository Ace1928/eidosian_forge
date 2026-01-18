from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ser', [Series(range(10), dtype=np.float64), Series([str(i) for i in range(10)], dtype=object)])
def test_squeeze_series_noop(self, ser):
    tm.assert_series_equal(ser.squeeze(), ser)