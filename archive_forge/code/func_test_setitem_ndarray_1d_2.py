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
def test_setitem_ndarray_1d_2(self):
    df = DataFrame(index=Index(np.arange(1, 11)))
    df['foo'] = np.zeros(10, dtype=np.float64)
    df['bar'] = np.zeros(10, dtype=complex)
    msg = 'Must have equal len keys and value when setting with an iterable'
    with pytest.raises(ValueError, match=msg):
        df[2:5] = np.arange(1, 4) * 1j