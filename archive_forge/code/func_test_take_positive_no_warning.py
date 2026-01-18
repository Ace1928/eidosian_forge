import numpy as np
import pytest
from pandas import Categorical
import pandas._testing as tm
def test_take_positive_no_warning(self):
    cat = Categorical(['a', 'b'])
    with tm.assert_produces_warning(None):
        cat.take([0, 0])