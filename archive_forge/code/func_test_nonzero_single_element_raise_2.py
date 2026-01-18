from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [np.nan, pd.NaT])
def test_nonzero_single_element_raise_2(self, data):
    msg_warn = 'Series.bool is now deprecated and will be removed in future version of pandas'
    msg_err = 'bool cannot act on a non-boolean single element Series'
    series = Series([data])
    with tm.assert_produces_warning(FutureWarning, match=msg_warn):
        with pytest.raises(ValueError, match=msg_err):
            series.bool()