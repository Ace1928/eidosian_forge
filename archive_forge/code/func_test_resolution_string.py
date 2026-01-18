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
def test_resolution_string(self):
    assert Timedelta(days=1).resolution_string == 'D'
    assert Timedelta(days=1, hours=6).resolution_string == 'h'
    assert Timedelta(days=1, minutes=6).resolution_string == 'min'
    assert Timedelta(days=1, seconds=6).resolution_string == 's'
    assert Timedelta(days=1, milliseconds=6).resolution_string == 'ms'
    assert Timedelta(days=1, microseconds=6).resolution_string == 'us'
    assert Timedelta(days=1, nanoseconds=6).resolution_string == 'ns'