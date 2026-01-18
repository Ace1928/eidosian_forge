from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_select_bad_cols(self):
    df = DataFrame([[1, 2]], columns=['A', 'B'])
    g = df.groupby('A')
    with pytest.raises(KeyError, match='"Columns not found: \'C\'"'):
        g[['C']]
    with pytest.raises(KeyError, match='^[^A]+$'):
        g[['A', 'C']]