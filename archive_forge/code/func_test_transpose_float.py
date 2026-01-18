import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_transpose_float(self, float_frame):
    frame = float_frame
    dft = frame.T
    for idx, series in dft.items():
        for col, value in series.items():
            if np.isnan(value):
                assert np.isnan(frame[col][idx])
            else:
                assert value == frame[col][idx]