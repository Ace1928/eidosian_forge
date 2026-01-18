import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_scope(self, engine, parser):
    skip_if_no_pandas_parser(parser)
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 2)), columns=list('ab'))
    a, b = (1, 2)
    res = df.query('a > b', engine=engine, parser=parser)
    expected = df[df.a > df.b]
    tm.assert_frame_equal(res, expected)
    res = df.query('@a > b', engine=engine, parser=parser)
    expected = df[a > df.b]
    tm.assert_frame_equal(res, expected)
    with pytest.raises(UndefinedVariableError, match="local variable 'c' is not defined"):
        df.query('@a > b > @c', engine=engine, parser=parser)
    with pytest.raises(UndefinedVariableError, match="name 'c' is not defined"):
        df.query('@a > b > c', engine=engine, parser=parser)