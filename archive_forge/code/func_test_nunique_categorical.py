import numpy as np
from pandas import (
def test_nunique_categorical():
    ser = Series(Categorical([]))
    assert ser.nunique() == 0
    ser = Series(Categorical([np.nan]))
    assert ser.nunique() == 0