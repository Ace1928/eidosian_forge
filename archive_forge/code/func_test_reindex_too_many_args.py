import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_too_many_args():
    ser = Series([1, 2])
    msg = 'reindex\\(\\) takes from 1 to 2 positional arguments but 3 were given'
    with pytest.raises(TypeError, match=msg):
        ser.reindex([2, 3], False)