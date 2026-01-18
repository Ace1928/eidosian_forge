import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.indexers import check_array_indexer
@pytest.mark.parametrize('indexer', [[0.0, 1.0], np.array([1.0, 2.0], dtype='float64'), np.array([True, False], dtype=object), pd.Index([True, False], dtype=object)])
def test_raise_invalid_array_dtypes(indexer):
    arr = np.array([1, 2, 3])
    msg = 'arrays used as indices must be of integer or boolean type'
    with pytest.raises(IndexError, match=msg):
        check_array_indexer(arr, indexer)