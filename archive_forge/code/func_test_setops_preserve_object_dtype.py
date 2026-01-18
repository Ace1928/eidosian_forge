from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
def test_setops_preserve_object_dtype(self):
    idx = Index([1, 2, 3], dtype=object)
    result = idx.intersection(idx[1:])
    expected = idx[1:]
    tm.assert_index_equal(result, expected)
    result = idx.intersection(idx[1:][::-1])
    tm.assert_index_equal(result, expected)
    result = idx._union(idx[1:], sort=None)
    expected = idx
    tm.assert_numpy_array_equal(result, expected.values)
    result = idx.union(idx[1:], sort=None)
    tm.assert_index_equal(result, expected)
    result = idx._union(idx[1:][::-1], sort=None)
    tm.assert_numpy_array_equal(result, expected.values)
    result = idx.union(idx[1:][::-1], sort=None)
    tm.assert_index_equal(result, expected)