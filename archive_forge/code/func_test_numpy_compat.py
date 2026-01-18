import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('func', ['cumprod', 'cumsum'])
def test_numpy_compat(func):
    df = DataFrame({'A': [1, 2, 1], 'B': [1, 2, 3]})
    g = df.groupby('A')
    msg = 'numpy operations are not valid with groupby'
    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(1, 2, 3)
    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(foo=1)