from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_timedelta_array_total_seconds(self):
    expected = Timedelta('2 min').total_seconds()
    result = pd.array([Timedelta('2 min')]).total_seconds()[0]
    assert result == expected