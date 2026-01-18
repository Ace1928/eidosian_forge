from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_margins_name_unicode(self):
    greek = 'Δοκιμή'
    frame = DataFrame({'foo': [1, 2, 3]}, columns=Index(['foo'], dtype=object))
    table = pivot_table(frame, index=['foo'], aggfunc=len, margins=True, margins_name=greek)
    index = Index([1, 2, 3, greek], dtype='object', name='foo')
    expected = DataFrame(index=index, columns=[])
    tm.assert_frame_equal(table, expected)