import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.indexes.common import Base
def test_view(self, simple_index):
    idx = simple_index
    idx_view = idx.view('i8')
    result = self._index_cls(idx)
    tm.assert_index_equal(result, idx)
    idx_view = idx.view(self._index_cls)
    result = self._index_cls(idx)
    tm.assert_index_equal(result, idx_view)