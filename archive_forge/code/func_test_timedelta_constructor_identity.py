from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_timedelta_constructor_identity():
    expected = Timedelta(np.timedelta64(1, 's'))
    result = Timedelta(expected)
    assert result is expected