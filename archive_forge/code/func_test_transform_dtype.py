import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_dtype():
    df = DataFrame([[1, 3], [2, 3]])
    result = df.groupby(1).transform('mean')
    expected = DataFrame([[1.5], [1.5]])
    tm.assert_frame_equal(result, expected)