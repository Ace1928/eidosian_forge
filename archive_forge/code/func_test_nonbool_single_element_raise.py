from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [1, 0, 'a', 0.0])
def test_nonbool_single_element_raise(self, data):
    msg_warn = 'Series.bool is now deprecated and will be removed in future version of pandas'
    msg_err1 = 'The truth value of a Series is ambiguous'
    msg_err2 = 'bool cannot act on a non-boolean single element Series'
    series = Series([data])
    with pytest.raises(ValueError, match=msg_err1):
        bool(series)
    with tm.assert_produces_warning(FutureWarning, match=msg_warn):
        with pytest.raises(ValueError, match=msg_err2):
            series.bool()