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
def test_repr_range_categories(self):
    rng = pd.Index(range(3))
    dtype = CategoricalDtype(categories=rng, ordered=False)
    result = repr(dtype)
    expected = 'CategoricalDtype(categories=range(0, 3), ordered=False, categories_dtype=int64)'
    assert result == expected