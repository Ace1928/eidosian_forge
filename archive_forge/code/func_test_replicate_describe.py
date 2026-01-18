import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_replicate_describe(string_series):
    expected = string_series.describe()
    result = string_series.apply({'count': 'count', 'mean': 'mean', 'std': 'std', 'min': 'min', '25%': lambda x: x.quantile(0.25), '50%': 'median', '75%': lambda x: x.quantile(0.75), 'max': 'max'})
    tm.assert_series_equal(result, expected)