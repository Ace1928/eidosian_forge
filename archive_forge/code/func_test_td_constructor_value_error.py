from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_td_constructor_value_error():
    msg = "Invalid type <class 'str'>. Must be int or float."
    with pytest.raises(TypeError, match=msg):
        Timedelta(nanoseconds='abc')