from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
def test_object_series_ok():

    class Dummy:

        def __init__(self, value) -> None:
            self.value = value

        def __add__(self, other):
            return self.value + other.value
    arr = np.array([Dummy(0), Dummy(1)])
    ser = pd.Series(arr)
    tm.assert_series_equal(np.add(ser, ser), pd.Series(np.add(ser, arr)))
    tm.assert_series_equal(np.add(ser, Dummy(1)), pd.Series(np.add(ser, Dummy(1))))