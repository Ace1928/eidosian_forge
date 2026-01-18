import numpy as np
import pytest
from pandas import Categorical
import pandas._testing as tm
def test_positional_take_unobserved(self, ordered):
    cat = Categorical(['a', 'b'], categories=['a', 'b', 'c'], ordered=ordered)
    result = cat.take([1, 0], allow_fill=False)
    expected = Categorical(['b', 'a'], categories=cat.categories, ordered=ordered)
    tm.assert_categorical_equal(result, expected)