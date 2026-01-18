from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('key', [Timestamp('1970-01-01'), Timestamp('1970-01-02'), datetime(1970, 1, 1), Timestamp('1970-01-03').to_datetime64(), np.datetime64('NaT')])
def test_timestamp_invalid_key(self, key):
    tdi = timedelta_range(0, periods=10)
    with pytest.raises(KeyError, match=re.escape(repr(key))):
        tdi.get_loc(key)