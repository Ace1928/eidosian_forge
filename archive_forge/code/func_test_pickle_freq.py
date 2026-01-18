import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_pickle_freq(self):
    prng = period_range('1/1/2011', '1/1/2012', freq='M')
    new_prng = tm.round_trip_pickle(prng)
    assert new_prng.freq == offsets.MonthEnd()
    assert new_prng.freqstr == 'M'