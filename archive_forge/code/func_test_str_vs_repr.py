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
def test_str_vs_repr(self, ordered, using_infer_string):
    c1 = CategoricalDtype(['a', 'b'], ordered=ordered)
    assert str(c1) == 'category'
    dtype = 'string' if using_infer_string else 'object'
    pat = f'CategoricalDtype\\(categories=\\[.*\\], ordered={{ordered}}, categories_dtype={dtype}\\)'
    assert re.match(pat.format(ordered=ordered), repr(c1))