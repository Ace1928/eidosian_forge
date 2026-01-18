from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
@pytest.mark.parametrize('freq', ['M', '2M', '3M'])
def test_period_addsub_nat(self, freq):
    per = Period('2011-01', freq=freq)
    assert NaT - per is NaT
    assert per - NaT is NaT
    assert NaT + per is NaT
    assert per + NaT is NaT