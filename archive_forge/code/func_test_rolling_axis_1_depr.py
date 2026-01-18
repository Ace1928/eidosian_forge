import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from .utils import (
def test_rolling_axis_1_depr():
    index = pandas.date_range('31/12/2000', periods=12, freq='min')
    data = {'A': range(12), 'B': range(12)}
    modin_df = pd.DataFrame(data, index=index)
    with pytest.warns(FutureWarning, match='Support for axis=1 in DataFrame.rolling is deprecated'):
        modin_df.rolling(window=3, axis=1)