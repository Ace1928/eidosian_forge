import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_numba_vs_python_noop(float_frame, apply_axis):
    func = lambda x: x
    result = float_frame.apply(func, engine='numba', axis=apply_axis)
    expected = float_frame.apply(func, engine='python', axis=apply_axis)
    tm.assert_frame_equal(result, expected)