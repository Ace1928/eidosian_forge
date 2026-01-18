import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_reindex_categorical_added_category(self):
    ci = CategoricalIndex([Interval(0, 1, closed='right'), Interval(1, 2, closed='right')], ordered=True)
    ci_add = CategoricalIndex([Interval(0, 1, closed='right'), Interval(1, 2, closed='right'), Interval(2, 3, closed='right'), Interval(3, 4, closed='right')], ordered=True)
    result, _ = ci.reindex(ci_add)
    expected = ci_add
    tm.assert_index_equal(expected, result)