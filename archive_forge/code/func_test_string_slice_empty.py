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
def test_string_slice_empty(self):
    df = DataFrame()
    assert not df.index._is_all_dates
    with pytest.raises(KeyError, match="'2011'"):
        df['2011']
    with pytest.raises(KeyError, match='^0$'):
        df.loc['2011', 0]