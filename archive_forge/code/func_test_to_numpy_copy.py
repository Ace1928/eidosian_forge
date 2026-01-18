import numpy as np
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_to_numpy_copy(self, using_copy_on_write):
    arr = np.random.default_rng(2).standard_normal((4, 3))
    df = DataFrame(arr)
    if using_copy_on_write:
        assert df.values.base is not arr
        assert df.to_numpy(copy=False).base is df.values.base
    else:
        assert df.values.base is arr
        assert df.to_numpy(copy=False).base is arr
    assert df.to_numpy(copy=True).base is not arr
    if using_copy_on_write:
        assert df.to_numpy(copy=False).base is df.values.base
    else:
        assert df.to_numpy(copy=False, na_value=np.nan).base is arr