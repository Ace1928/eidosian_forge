import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_axis_raises(self):
    df = DataFrame({'A': [1, 2], 'B': [1, 2]})
    with pytest.raises(ValueError, match='Use `.rename`'):
        df.rename_axis(id, axis=0)
    with pytest.raises(ValueError, match='Use `.rename`'):
        df.rename_axis({0: 10, 1: 20}, axis=0)
    with pytest.raises(ValueError, match='Use `.rename`'):
        df.rename_axis(id, axis=1)
    with pytest.raises(ValueError, match='Use `.rename`'):
        df['A'].rename_axis(id)