from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
@pytest.mark.parametrize('null', [pd.NaT, pd.NaT.to_numpy('M8[ns]'), pd.NaT.to_numpy('m8[ns]')])
def test_setting_mismatched_na_into_nullable_fails(self, null, any_numeric_ea_dtype):
    df = DataFrame({'A': [1, 2, 3]}, dtype=any_numeric_ea_dtype)
    ser = df['A'].copy()
    arr = ser._values
    msg = '|'.join(['timedelta64\\[ns\\] cannot be converted to (Floating|Integer)Dtype', 'datetime64\\[ns\\] cannot be converted to (Floating|Integer)Dtype', "'values' contains non-numeric NA", "Invalid value '.*' for dtype (U?Int|Float)\\d{1,2}"])
    with pytest.raises(TypeError, match=msg):
        arr[0] = null
    with pytest.raises(TypeError, match=msg):
        arr[:2] = [null, null]
    with pytest.raises(TypeError, match=msg):
        ser[0] = null
    with pytest.raises(TypeError, match=msg):
        ser[:2] = [null, null]
    with pytest.raises(TypeError, match=msg):
        ser.iloc[0] = null
    with pytest.raises(TypeError, match=msg):
        ser.iloc[:2] = [null, null]
    with pytest.raises(TypeError, match=msg):
        df.iloc[0, 0] = null
    with pytest.raises(TypeError, match=msg):
        df.iloc[:2, 0] = [null, null]
    df2 = df.copy()
    df2['B'] = ser.copy()
    with pytest.raises(TypeError, match=msg):
        df2.iloc[0, 0] = null
    with pytest.raises(TypeError, match=msg):
        df2.iloc[:2, 0] = [null, null]