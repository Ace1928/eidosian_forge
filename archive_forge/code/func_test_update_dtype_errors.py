import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('bad_dtype', ['foo', object, np.int64, PeriodDtype('Q')])
def test_update_dtype_errors(self, bad_dtype):
    dtype = CategoricalDtype(list('abc'), False)
    msg = 'a CategoricalDtype must be passed to perform an update, '
    with pytest.raises(ValueError, match=msg):
        dtype.update_dtype(bad_dtype)