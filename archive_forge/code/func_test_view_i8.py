import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_view_i8(self):
    ci = CategoricalIndex(list('ab') * 50)
    msg = 'When changing to a larger dtype, its size must be a divisor'
    with pytest.raises(ValueError, match=msg):
        ci.view('i8')
    with pytest.raises(ValueError, match=msg):
        ci._data.view('i8')
    ci = ci[:-4]
    res = ci.view('i8')
    expected = ci._data.codes.view('i8')
    tm.assert_numpy_array_equal(res, expected)
    cat = ci._data
    tm.assert_numpy_array_equal(cat.view('i8'), expected)