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
@pytest.mark.parametrize('op,targop', [('mean', np.mean), ('median', np.median), ('std', np.std), ('var', np.var), ('sum', np.sum), ('prod', np.prod), ('min', np.min), ('max', np.max), ('first', lambda x: x.iloc[0]), ('last', lambda x: x.iloc[-1]), ('count', np.size), pytest.param('sem', scipy_sem, marks=td.skip_if_no_scipy)])
def test_ops_general(op, targop):
    df = DataFrame(np.random.randn(1000))
    labels = np.random.randint(0, 50, size=1000).astype(float)
    result = getattr(df.groupby(labels), op)()
    expected = df.groupby(labels).agg(targop)
    tm.assert_frame_equal(result, expected)