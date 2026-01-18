import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_array2():
    arr = np.random.RandomState(0).randint(0, 5, size=(10, 3), dtype=np.int64)
    df = DataFrame(arr, columns=list('ABC'))
    result = df.groupby('A').quantile([0.3, 0.7])
    expected = DataFrame({'B': [0.9, 2.1, 2.2, 3.4, 1.6, 2.4, 2.3, 2.7, 0.0, 0.0], 'C': [1.2, 2.8, 1.8, 3.0, 0.0, 0.0, 1.9, 3.1, 3.0, 3.0]}, index=pd.MultiIndex.from_product([[0, 1, 2, 3, 4], [0.3, 0.7]], names=['A', None]))
    tm.assert_frame_equal(result, expected)