import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_table_series_valueerror(self):

    def f(x):
        return np.sum(x, axis=0) + 1
    with pytest.raises(ValueError, match="method='table' not applicable for Series objects."):
        Series(range(1)).rolling(1, method='table').apply(f, engine='numba', raw=True)