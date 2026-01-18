from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_rolling_wrong_param_min_period():
    name_l = ['Alice'] * 5 + ['Bob'] * 5
    val_l = [np.nan, np.nan, 1, 2, 3] + [np.nan, 1, 2, 3, 4]
    test_df = DataFrame([name_l, val_l]).T
    test_df.columns = ['name', 'val']
    result_error_msg = "__init__\\(\\) got an unexpected keyword argument 'min_period'"
    with pytest.raises(TypeError, match=result_error_msg):
        test_df.groupby('name')['val'].rolling(window=2, min_period=1).sum()