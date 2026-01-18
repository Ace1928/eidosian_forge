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
@pytest.mark.skip_ubsan
def test_out_of_s_bounds_datetime64(self, constructor):
    scalar = np.datetime64(np.iinfo(np.int64).max, 'D')
    result = constructor(scalar)
    item = get1(result)
    assert type(item) is np.datetime64
    dtype = tm.get_dtype(result)
    assert dtype == object