import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_start_end_non_nat(self):
    msg = 'start and end must not be NaT'
    with pytest.raises(ValueError, match=msg):
        period_range(start=NaT, end='2018Q1')
    with pytest.raises(ValueError, match=msg):
        period_range(start=NaT, end='2018Q1', freq='Q')
    with pytest.raises(ValueError, match=msg):
        period_range(start='2017Q1', end=NaT)
    with pytest.raises(ValueError, match=msg):
        period_range(start='2017Q1', end=NaT, freq='Q')