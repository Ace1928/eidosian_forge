from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nonzero_single_element(self):
    msg_warn = 'Series.bool is now deprecated and will be removed in future version of pandas'
    ser = Series([True])
    ser1 = Series([False])
    with tm.assert_produces_warning(FutureWarning, match=msg_warn):
        assert ser.bool()
    with tm.assert_produces_warning(FutureWarning, match=msg_warn):
        assert not ser1.bool()