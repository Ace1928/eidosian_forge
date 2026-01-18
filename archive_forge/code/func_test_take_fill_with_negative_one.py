import numpy as np
import pytest
from pandas import Categorical
import pandas._testing as tm
def test_take_fill_with_negative_one(self):
    cat = Categorical([-1, 0, 1])
    result = cat.take([0, -1, 1], allow_fill=True, fill_value=-1)
    expected = Categorical([-1, -1, 0], categories=[-1, 0, 1])
    tm.assert_categorical_equal(result, expected)