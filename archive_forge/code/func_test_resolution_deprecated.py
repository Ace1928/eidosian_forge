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
def test_resolution_deprecated(self):
    td = Timedelta(days=4, hours=3)
    result = td.resolution
    assert result == Timedelta(nanoseconds=1)
    result = Timedelta.resolution
    assert result == Timedelta(nanoseconds=1)