import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_series_with_name_not_matching_column(self):
    x = Series(range(5), name=1)
    y = Series(range(5), name=0)
    result = DataFrame(x, columns=[0])
    expected = DataFrame([], columns=[0])
    tm.assert_frame_equal(result, expected)
    result = DataFrame(y, columns=[1])
    expected = DataFrame([], columns=[1])
    tm.assert_frame_equal(result, expected)