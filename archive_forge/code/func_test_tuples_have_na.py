import numpy as np
import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_tuples_have_na():
    index = MultiIndex(levels=[[1, 0], [0, 1, 2, 3]], codes=[[1, 1, 1, 1, -1, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]])
    assert pd.isna(index[4][0])
    assert pd.isna(index.values[4][0])