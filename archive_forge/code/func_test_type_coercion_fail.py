import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_type_coercion_fail(self, any_int_numpy_dtype):
    msg = 'Trying to coerce float values to integers'
    with pytest.raises(ValueError, match=msg):
        Index([1, 2, 3.5], dtype=any_int_numpy_dtype)