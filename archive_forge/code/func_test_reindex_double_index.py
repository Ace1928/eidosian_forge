import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_double_index():
    ser = Series([1, 2])
    msg = "reindex\\(\\) got multiple values for argument 'index'"
    with pytest.raises(TypeError, match=msg):
        ser.reindex([2, 3], index=[3, 4])