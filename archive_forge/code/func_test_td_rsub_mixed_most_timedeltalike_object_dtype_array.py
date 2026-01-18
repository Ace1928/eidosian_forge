from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rsub_mixed_most_timedeltalike_object_dtype_array(self):
    now = Timestamp('2021-11-09 09:54:00')
    arr = np.array([now, Timedelta('1D'), np.timedelta64(2, 'h')])
    msg = "unsupported operand type\\(s\\) for \\-: 'Timedelta' and 'Timestamp'"
    with pytest.raises(TypeError, match=msg):
        Timedelta('1D') - arr