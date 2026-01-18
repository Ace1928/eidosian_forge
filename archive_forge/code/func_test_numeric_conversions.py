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
def test_numeric_conversions(self):
    assert Timedelta(0) == np.timedelta64(0, 'ns')
    assert Timedelta(10) == np.timedelta64(10, 'ns')
    assert Timedelta(10, unit='ns') == np.timedelta64(10, 'ns')
    assert Timedelta(10, unit='us') == np.timedelta64(10, 'us')
    assert Timedelta(10, unit='ms') == np.timedelta64(10, 'ms')
    assert Timedelta(10, unit='s') == np.timedelta64(10, 's')
    assert Timedelta(10, unit='d') == np.timedelta64(10, 'D')