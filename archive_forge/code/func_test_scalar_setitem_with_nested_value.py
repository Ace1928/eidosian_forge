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
@pytest.mark.parametrize('value', [(0, 1), [0, 1], np.array([0, 1]), array.array('b', [0, 1])])
def test_scalar_setitem_with_nested_value(value):
    df = DataFrame({'A': [1, 2, 3]})
    msg = '|'.join(['Must have equal len keys and value', 'setting an array element with a sequence'])
    with pytest.raises(ValueError, match=msg):
        df.loc[0, 'B'] = value
    df = DataFrame({'A': [1, 2, 3], 'B': np.array([1, 'a', 'b'], dtype=object)})
    with pytest.raises(ValueError, match='Must have equal len keys and value'):
        df.loc[0, 'B'] = value