from __future__ import annotations
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
def test_validate_allhashable():
    assert com.validate_all_hashable(1, 'a') is None
    with pytest.raises(TypeError, match='All elements must be hashable'):
        com.validate_all_hashable([])
    with pytest.raises(TypeError, match='list must be a hashable type'):
        com.validate_all_hashable([], error_name='list')