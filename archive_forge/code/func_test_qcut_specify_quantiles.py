import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
def test_qcut_specify_quantiles():
    arr = np.random.default_rng(2).standard_normal(100)
    factor = qcut(arr, [0, 0.25, 0.5, 0.75, 1.0])
    expected = qcut(arr, 4)
    tm.assert_categorical_equal(factor, expected)