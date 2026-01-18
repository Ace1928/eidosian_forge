from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_squeeze_frame_noop(self):
    df = DataFrame(np.eye(2))
    tm.assert_frame_equal(df.squeeze(), df)