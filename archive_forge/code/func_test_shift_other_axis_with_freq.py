import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_other_axis_with_freq(self, datetime_frame):
    obj = datetime_frame.T
    offset = offsets.BDay()
    shifted = obj.shift(5, freq=offset, axis=1)
    assert len(shifted) == len(obj)
    unshifted = shifted.shift(-5, freq=offset, axis=1)
    tm.assert_equal(unshifted, obj)