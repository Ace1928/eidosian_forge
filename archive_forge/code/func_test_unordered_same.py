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
@pytest.mark.parametrize('ordered', [False, None])
def test_unordered_same(self, ordered):
    c1 = CategoricalDtype(['a', 'b'], ordered=ordered)
    c2 = CategoricalDtype(['b', 'a'], ordered=ordered)
    assert hash(c1) == hash(c2)