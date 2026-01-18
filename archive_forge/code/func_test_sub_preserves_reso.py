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
def test_sub_preserves_reso(self, td, unit):
    res = td - td
    expected = Timedelta._from_value_and_reso(0, unit)
    assert res == expected
    assert res._creso == unit