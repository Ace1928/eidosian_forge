from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import iNaT
from pandas.errors import OutOfBoundsTimedelta
from pandas import Timedelta
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_round_non_nano(self, unit):
    td = Timedelta('1 days 02:34:57').as_unit(unit)
    res = td.round('min')
    assert res == Timedelta('1 days 02:35:00')
    assert res._creso == td._creso
    res = td.floor('min')
    assert res == Timedelta('1 days 02:34:00')
    assert res._creso == td._creso
    res = td.ceil('min')
    assert res == Timedelta('1 days 02:35:00')
    assert res._creso == td._creso