import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_scalar_series(self, data, box_in_series):
    if box_in_series:
        data = pd.Series(data)
    data[0] = data[1]
    assert data[0] == data[1]