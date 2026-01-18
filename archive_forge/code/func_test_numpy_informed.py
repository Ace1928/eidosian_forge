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
def test_numpy_informed(self, dtype):
    msg = '|'.join(['data type not understood', "Cannot interpret '.*' as a data type"])
    with pytest.raises(TypeError, match=msg):
        np.dtype(dtype)
    assert not dtype == np.str_
    assert not np.str_ == dtype