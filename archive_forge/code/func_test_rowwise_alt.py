import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_rowwise_alt(self):
    df = DataFrame({0: [0, 0.5, 1.0, np.nan, 4, 8, np.nan, np.nan, 64], 1: [1, 2, 3, 4, 3, 2, 1, 0, -1]})
    df.interpolate(axis=0)