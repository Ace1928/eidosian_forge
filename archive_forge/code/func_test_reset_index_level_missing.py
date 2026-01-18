from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx_lev', [['A', 'B'], ['A']])
def test_reset_index_level_missing(self, idx_lev):
    df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=['A', 'B', 'C', 'D'])
    with pytest.raises(KeyError, match='(L|l)evel \\(?E\\)?'):
        df.set_index(idx_lev).reset_index(level=['A', 'E'])
    with pytest.raises(IndexError, match='Too many levels'):
        df.set_index(idx_lev).reset_index(level=[0, 1, 2])