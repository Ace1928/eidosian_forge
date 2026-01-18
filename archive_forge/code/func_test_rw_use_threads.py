import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.feather_format import read_feather, to_feather  # isort:skip
def test_rw_use_threads(self):
    df = pd.DataFrame({'A': np.arange(100000)})
    self.check_round_trip(df, use_threads=True)
    self.check_round_trip(df, use_threads=False)