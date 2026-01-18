from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize(('input', 'expected'), [('8:53:08.71800000001', '8:53:08.718'), ('8:53:08.718001', '8:53:08.718001'), ('8:53:08.7180000001', '8:53:08.7180000001'), ('-8:53:08.71800000001', '-8:53:08.718'), ('8:53:08.7180000089', '8:53:08.718000008')])
@pytest.mark.parametrize('func', [pd.Timedelta, to_timedelta])
def test_to_timedelta_precision_over_nanos(self, input, expected, func):
    expected = pd.Timedelta(expected)
    result = func(input)
    assert result == expected