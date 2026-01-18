from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_drop_infer_string(self):
    pytest.importorskip('pyarrow')
    ser = Series(['a', 'b', 'c'], dtype=object)
    with option_context('future.infer_string', True):
        result = ser.reset_index(drop=True)
    tm.assert_series_equal(result, ser)