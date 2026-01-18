from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
def test_object_dtype_ok():

    class Thing:

        def __init__(self, value) -> None:
            self.value = value

        def __add__(self, other):
            other = getattr(other, 'value', other)
            return type(self)(self.value + other)

        def __eq__(self, other) -> bool:
            return type(other) is Thing and self.value == other.value

        def __repr__(self) -> str:
            return f'Thing({self.value})'
    s = pd.Series([Thing(1), Thing(2)])
    result = np.add(s, Thing(1))
    expected = pd.Series([Thing(2), Thing(3)])
    tm.assert_series_equal(result, expected)