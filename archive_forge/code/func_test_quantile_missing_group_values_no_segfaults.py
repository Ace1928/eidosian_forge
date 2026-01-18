import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_missing_group_values_no_segfaults():
    data = np.array([1.0, np.nan, 1.0])
    df = DataFrame({'key': data, 'val': range(3)})
    grp = df.groupby('key')
    for _ in range(100):
        grp.quantile()