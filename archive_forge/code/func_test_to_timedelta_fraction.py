from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_fraction(self):
    result = to_timedelta(1.0 / 3, unit='h')
    expected = pd.Timedelta('0 days 00:19:59.999999998')
    assert result == expected