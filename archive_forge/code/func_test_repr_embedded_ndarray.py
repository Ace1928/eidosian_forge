from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_repr_embedded_ndarray(self):
    arr = np.empty(10, dtype=[('err', object)])
    for i in range(len(arr)):
        arr['err'][i] = np.random.default_rng(2).standard_normal(i)
    df = DataFrame(arr)
    repr(df['err'])
    repr(df)
    df.to_string()