import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_none_to_nan(cls, dtype):
    a = cls._from_sequence(['a', None, 'b'], dtype=dtype)
    assert a[1] is not None
    assert a[1] is na_val(a.dtype)