import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.feather_format import read_feather, to_feather  # isort:skip
def test_passthrough_keywords(self):
    df = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD'), dtype=object), index=pd.Index([f'i-{i}' for i in range(30)], dtype=object)).reset_index()
    self.check_round_trip(df, write_kwargs={'version': 1})