import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.indexes.common import Base
def test_shift_empty(self, simple_index):
    idx = simple_index[:0]
    tm.assert_index_equal(idx, idx.shift(1))