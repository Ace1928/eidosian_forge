import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_empty_indexer(self, data, box_in_series):
    if box_in_series:
        data = pd.Series(data)
    original = data.copy()
    data[np.array([], dtype=int)] = []
    tm.assert_equal(data, original)