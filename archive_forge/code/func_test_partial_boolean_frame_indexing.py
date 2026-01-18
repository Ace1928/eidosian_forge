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
def test_partial_boolean_frame_indexing(self):
    df = DataFrame(np.arange(9.0).reshape(3, 3), index=list('abc'), columns=list('ABC'))
    index_df = DataFrame(1, index=list('ab'), columns=list('AB'))
    result = df[index_df.notnull()]
    expected = DataFrame(np.array([[0.0, 1.0, np.nan], [3.0, 4.0, np.nan], [np.nan] * 3]), index=list('abc'), columns=list('ABC'))
    tm.assert_frame_equal(result, expected)