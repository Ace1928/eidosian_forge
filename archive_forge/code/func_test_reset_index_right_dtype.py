from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_right_dtype(self):
    time = np.arange(0.0, 10, np.sqrt(2) / 2)
    s1 = Series(9.81 * time ** 2 / 2, index=Index(time, name='time'), name='speed')
    df = DataFrame(s1)
    reset = s1.reset_index()
    assert reset['time'].dtype == np.float64
    reset = df.reset_index()
    assert reset['time'].dtype == np.float64