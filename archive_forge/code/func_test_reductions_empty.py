import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize('name', ['std', 'min', 'max', 'median', 'mean'])
@pytest.mark.parametrize('skipna', [True, False])
def test_reductions_empty(self, name, skipna):
    tdi = pd.TimedeltaIndex([])
    arr = tdi.array
    result = getattr(tdi, name)(skipna=skipna)
    assert result is pd.NaT
    result = getattr(arr, name)(skipna=skipna)
    assert result is pd.NaT