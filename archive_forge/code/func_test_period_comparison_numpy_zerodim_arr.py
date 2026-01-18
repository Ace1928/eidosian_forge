from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
@pytest.mark.parametrize('zerodim_arr, expected', ((np.array(0), False), (np.array(Period('2000-01', 'M')), True)))
def test_period_comparison_numpy_zerodim_arr(self, zerodim_arr, expected):
    per = Period('2000-01', 'M')
    assert (per == zerodim_arr) is expected
    assert (zerodim_arr == per) is expected