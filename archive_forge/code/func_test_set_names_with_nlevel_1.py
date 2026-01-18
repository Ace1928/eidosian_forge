import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('inplace', [True, False])
def test_set_names_with_nlevel_1(inplace):
    expected = MultiIndex(levels=[[0, 1]], codes=[[0, 1]], names=['first'])
    m = MultiIndex.from_product([[0, 1]])
    result = m.set_names('first', level=0, inplace=inplace)
    if inplace:
        result = m
    tm.assert_index_equal(result, expected)