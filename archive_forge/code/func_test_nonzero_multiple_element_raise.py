from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [(True, True), (False, False)])
def test_nonzero_multiple_element_raise(self, data):
    msg_warn = 'Series.bool is now deprecated and will be removed in future version of pandas'
    msg_err = 'The truth value of a Series is ambiguous'
    series = Series([data])
    with pytest.raises(ValueError, match=msg_err):
        bool(series)
    with tm.assert_produces_warning(FutureWarning, match=msg_warn):
        with pytest.raises(ValueError, match=msg_err):
            series.bool()