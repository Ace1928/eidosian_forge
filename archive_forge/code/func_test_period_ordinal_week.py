import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.period import (
import pandas._testing as tm
@pytest.mark.parametrize('dt,expected', [((1970, 1, 4, 0, 0, 0, 0, 0), 1), ((1970, 1, 5, 0, 0, 0, 0, 0), 2), ((2013, 10, 6, 0, 0, 0, 0, 0), 2284), ((2013, 10, 7, 0, 0, 0, 0, 0), 2285)])
def test_period_ordinal_week(dt, expected):
    args = dt + (get_freq_code('W'),)
    assert period_ordinal(*args) == expected