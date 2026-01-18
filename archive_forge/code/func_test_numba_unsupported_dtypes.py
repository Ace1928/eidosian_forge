import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_numba_unsupported_dtypes(apply_axis):
    f = lambda x: x
    df = DataFrame({'a': [1, 2], 'b': ['a', 'b'], 'c': [4, 5]})
    df['c'] = df['c'].astype('double[pyarrow]')
    with pytest.raises(ValueError, match="Column b must have a numeric dtype. Found 'object|string' instead"):
        df.apply(f, engine='numba', axis=apply_axis)
    with pytest.raises(ValueError, match='Column c is backed by an extension array, which is not supported by the numba engine.'):
        df['c'].to_frame().apply(f, engine='numba', axis=apply_axis)