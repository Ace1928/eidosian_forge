from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_add_timestamp_overflow(self):
    ts = Timestamp('1700-01-01').as_unit('ns')
    msg = "Cannot cast 259987 from D to 'ns' without overflow."
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        ts + Timedelta(13 * 19999, unit='D')
    msg = "Cannot cast 259987 days 00:00:00 to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        ts + timedelta(days=13 * 19999)