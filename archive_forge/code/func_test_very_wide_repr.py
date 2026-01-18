from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_very_wide_repr(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 20)), columns=np.array(['a' * 10] * 20, dtype=object))
    repr(df)