from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
def test_to_numpy_alias(self):
    td = Timedelta('10m7s')
    assert td.to_timedelta64() == td.to_numpy()
    msg = 'dtype and copy arguments are ignored'
    with pytest.raises(ValueError, match=msg):
        td.to_numpy('m8[s]')
    with pytest.raises(ValueError, match=msg):
        td.to_numpy(copy=True)