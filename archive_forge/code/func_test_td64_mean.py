import inspect
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [Series, pd.Index, pd.array])
def test_td64_mean(self, box):
    m8values = np.array([0, 3, -2, -7, 1, 2, -1, 3, 5, -2, 4], 'm8[D]')
    tdi = pd.TimedeltaIndex(m8values).as_unit('ns')
    tdarr = tdi._data
    obj = box(tdarr, copy=False)
    result = obj.mean()
    expected = np.array(tdarr).mean()
    assert result == expected
    tdarr[0] = pd.NaT
    assert obj.mean(skipna=False) is pd.NaT
    result2 = obj.mean(skipna=True)
    assert result2 == tdi[1:].mean()
    assert result2.round('us') == (result * 11.0 / 10).round('us')