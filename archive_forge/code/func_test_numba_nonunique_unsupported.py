import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_numba_nonunique_unsupported(apply_axis):
    f = lambda x: x
    df = DataFrame({'a': [1, 2]}, index=Index(['a', 'a']))
    with pytest.raises(NotImplementedError, match="The index/columns must be unique when raw=False and engine='numba'"):
        df.apply(f, engine='numba', axis=apply_axis)