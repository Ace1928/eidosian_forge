import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_nuisance_raises(df):
    df.columns = ['A', 'B', 'B', 'D']
    grouped = df.groupby('A')
    gbc = grouped['B']
    with pytest.raises(TypeError, match='Could not convert'):
        gbc.transform(lambda x: np.mean(x))
    with pytest.raises(TypeError, match='Could not convert'):
        df.groupby('A').transform(lambda x: np.mean(x))