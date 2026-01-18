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
def test_timedelta_class_min_max_resolution():
    assert Timedelta.min == Timedelta(NaT._value + 1)
    assert Timedelta.min._creso == NpyDatetimeUnit.NPY_FR_ns.value
    assert Timedelta.max == Timedelta(np.iinfo(np.int64).max)
    assert Timedelta.max._creso == NpyDatetimeUnit.NPY_FR_ns.value
    assert Timedelta.resolution == Timedelta(1)
    assert Timedelta.resolution._creso == NpyDatetimeUnit.NPY_FR_ns.value