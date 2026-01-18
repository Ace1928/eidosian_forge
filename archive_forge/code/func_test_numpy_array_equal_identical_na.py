import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_numpy_array_equal_identical_na(nulls_fixture):
    a = np.array([nulls_fixture], dtype=object)
    tm.assert_numpy_array_equal(a, a)
    if hasattr(nulls_fixture, 'copy'):
        other = nulls_fixture.copy()
    else:
        other = copy.copy(nulls_fixture)
    b = np.array([other], dtype=object)
    tm.assert_numpy_array_equal(a, b)