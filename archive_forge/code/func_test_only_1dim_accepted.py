import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_only_1dim_accepted(self):
    arr = np.array([0, 1, 2, 3], dtype='M8[h]').astype('M8[ns]')
    depr_msg = 'DatetimeArray.__init__ is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(ValueError, match='Only 1-dimensional'):
            DatetimeArray(arr.reshape(2, 2, 1))
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(ValueError, match='Only 1-dimensional'):
            DatetimeArray(arr[[0]].squeeze())