from datetime import time
import locale
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import Series
import pandas._testing as tm
from pandas.core.tools.times import to_time
def test_odd_format(self):
    new_string = '14.15'
    msg = "Cannot convert arg \\['14\\.15'\\] to a time"
    if not PY311:
        with pytest.raises(ValueError, match=msg):
            to_time(new_string)
    assert to_time(new_string, format='%H.%M') == time(14, 15)