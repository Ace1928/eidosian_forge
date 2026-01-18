from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_setitem_from_duplicate_axis(self):
    df = DataFrame([[20, 'a'], [200, 'a'], [200, 'a']], columns=['col1', 'col2'], index=[10, 1, 1])
    df.loc[1, 'col1'] = np.arange(2)
    expected = DataFrame([[20, 'a'], [0, 'a'], [1, 'a']], columns=['col1', 'col2'], index=[10, 1, 1])
    tm.assert_frame_equal(df, expected)