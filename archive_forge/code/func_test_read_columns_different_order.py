import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.feather_format import read_feather, to_feather  # isort:skip
def test_read_columns_different_order(self):
    df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y'], 'C': [True, False]})
    expected = df[['B', 'A']]
    self.check_round_trip(df, expected, columns=['B', 'A'])