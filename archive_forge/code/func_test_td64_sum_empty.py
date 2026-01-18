import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('skipna', [True, False])
def test_td64_sum_empty(skipna):
    ser = Series([], dtype='timedelta64[ns]')
    result = ser.sum(skipna=skipna)
    assert isinstance(result, pd.Timedelta)
    assert result == pd.Timedelta(0)