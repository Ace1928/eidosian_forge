import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
def test_numpy_unique(datetime_series):
    np.unique(datetime_series)