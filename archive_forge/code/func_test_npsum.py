import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays import TimedeltaArray
def test_npsum(self):
    tdi = pd.TimedeltaIndex(['3h', '3h', '2h', '5h', '4h'])
    arr = tdi.array
    result = np.sum(tdi)
    expected = Timedelta(hours=17)
    assert isinstance(result, Timedelta)
    assert result == expected
    result = np.sum(arr)
    assert isinstance(result, Timedelta)
    assert result == expected