from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_sub_timedelta64(self):
    td = Timedelta(10, unit='d')
    expected = Timedelta(0, unit='ns')
    result = td - td.to_timedelta64()
    assert isinstance(result, Timedelta)
    assert result == expected
    result = td.to_timedelta64() - td
    assert isinstance(result, Timedelta)
    assert result == expected