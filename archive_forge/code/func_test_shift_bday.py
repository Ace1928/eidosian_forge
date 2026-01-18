from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq', ['B', 'C'])
def test_shift_bday(self, freq, unit):
    rng = date_range(START, END, freq=freq, unit=unit)
    shifted = rng.shift(5)
    assert shifted[0] == rng[5]
    assert shifted.freq == rng.freq
    shifted = rng.shift(-5)
    assert shifted[5] == rng[0]
    assert shifted.freq == rng.freq
    shifted = rng.shift(0)
    assert shifted[0] == rng[0]
    assert shifted.freq == rng.freq