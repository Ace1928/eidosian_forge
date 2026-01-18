import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('dtype', ['int32', 'int64', 'bool'])
@pytest.mark.parametrize('box', [True, False], ids=['series', 'array'])
def test_to_numpy_na_raises(box, dtype):
    con = pd.Series if box else pd.array
    arr = con([0.0, 1.0, None], dtype='Float64')
    with pytest.raises(ValueError, match=dtype):
        arr.to_numpy(dtype=dtype)