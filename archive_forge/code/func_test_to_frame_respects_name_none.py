import pytest
from pandas import (
import pandas._testing as tm
def test_to_frame_respects_name_none(self):
    ser = Series(range(3))
    result = ser.to_frame(None)
    exp_index = Index([None], dtype=object)
    tm.assert_index_equal(result.columns, exp_index)
    result = ser.rename('foo').to_frame(None)
    exp_index = Index([None], dtype=object)
    tm.assert_index_equal(result.columns, exp_index)