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
def test_total_seconds_precision(self):
    assert Timedelta('30s').total_seconds() == 30.0
    assert Timedelta('0').total_seconds() == 0.0
    assert Timedelta('-2s').total_seconds() == -2.0
    assert Timedelta('5.324s').total_seconds() == 5.324
    assert Timedelta('30s').total_seconds() - 30.0 < 1e-20
    assert 30.0 - Timedelta('30s').total_seconds() < 1e-20