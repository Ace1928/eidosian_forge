import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_take_sequence(self, data):
    result = pd.Series(data)[[0, 1, 3]]
    assert result.iloc[0] == data[0]
    assert result.iloc[1] == data[1]
    assert result.iloc[2] == data[3]