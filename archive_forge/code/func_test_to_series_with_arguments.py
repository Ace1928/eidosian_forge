import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
def test_to_series_with_arguments(self, index):
    ser = index.to_series(index=index)
    assert ser.values is not index.values
    assert ser.index is index
    assert ser.name == index.name
    ser = index.to_series(name='__test')
    assert ser.values is not index.values
    assert ser.index is not index
    assert ser.name != index.name