from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_timedelta_new_npnat():
    nat = np.timedelta64('NaT', 'h')
    assert Timedelta(nat) is NaT