import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_iloc_scalar_single(self, data):
    df = pd.DataFrame({'B': data})
    df.iloc[10, 0] = data[1]
    assert df.loc[10, 'B'] == data[1]