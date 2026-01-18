from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_floordiv_invalid_scalar(self):
    td = Timedelta(hours=3, minutes=4)
    msg = '|'.join(['Invalid dtype datetime64\\[D\\] for __floordiv__', "'dtype' is an invalid keyword argument for this function", "ufunc '?floor_divide'? cannot use operands with types"])
    with pytest.raises(TypeError, match=msg):
        td // np.datetime64('2016-01-01', dtype='datetime64[us]')