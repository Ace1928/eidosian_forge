import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_none_coercion_mixed_dtypes(self):
    start_dataframe = DataFrame({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0], 'c': [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], 'd': ['a', 'b', 'c']})
    start_dataframe.iloc[0] = None
    exp = DataFrame({'a': [np.nan, 2, 3], 'b': [np.nan, 2.0, 3.0], 'c': [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)], 'd': [None, 'b', 'c']})
    tm.assert_frame_equal(start_dataframe, exp)