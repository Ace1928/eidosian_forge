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
@pytest.mark.parametrize('dtype, expected', [('int64', None), ('interval', IntervalDtype()), ('interval[int64, neither]', IntervalDtype()), ('interval[datetime64[ns], left]', IntervalDtype('datetime64[ns]', 'left')), ('period[D]', PeriodDtype('D')), ('category', CategoricalDtype()), ('datetime64[ns, US/Eastern]', DatetimeTZDtype('ns', 'US/Eastern'))])
def test_registry_find(dtype, expected):
    assert registry.find(dtype) == expected