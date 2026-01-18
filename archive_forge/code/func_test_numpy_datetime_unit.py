import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_numpy_datetime_unit(self, unit):
    data = np.array([1, 2, 3], dtype=f'M8[{unit}]')
    arr = DatetimeArray._from_sequence(data)
    assert arr.unit == unit
    assert arr[0].unit == unit