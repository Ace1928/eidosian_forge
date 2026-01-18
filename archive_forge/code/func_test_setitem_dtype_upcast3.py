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
def test_setitem_dtype_upcast3(self):
    left = DataFrame(np.arange(6, dtype='int64').reshape(2, 3) / 10.0, index=list('ab'), columns=['foo', 'bar', 'baz'])
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        left.loc['a', 'bar'] = 'wxyz'
    right = DataFrame([[0, 'wxyz', 0.2], [0.3, 0.4, 0.5]], index=list('ab'), columns=['foo', 'bar', 'baz'])
    tm.assert_frame_equal(left, right)
    assert is_float_dtype(left['foo'])
    assert is_float_dtype(left['baz'])