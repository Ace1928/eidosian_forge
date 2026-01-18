import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_transpose_get_view(self, float_frame, using_copy_on_write):
    dft = float_frame.T
    dft.iloc[:, 5:10] = 5
    if using_copy_on_write:
        assert (float_frame.values[5:10] != 5).all()
    else:
        assert (float_frame.values[5:10] == 5).all()