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
def test_order_matters(self):
    categories = ['a', 'b']
    c1 = CategoricalDtype(categories, ordered=True)
    c2 = CategoricalDtype(categories, ordered=False)
    c3 = CategoricalDtype(categories, ordered=None)
    assert c1 is not c2
    assert c1 is not c3