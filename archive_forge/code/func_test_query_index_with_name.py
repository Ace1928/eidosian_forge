import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_index_with_name(self, engine, parser):
    df = DataFrame(np.random.default_rng(2).integers(10, size=(10, 3)), index=Index(range(10), name='blob'), columns=['a', 'b', 'c'])
    res = df.query('(blob < 5) & (a < b)', engine=engine, parser=parser)
    expec = df[(df.index < 5) & (df.a < df.b)]
    tm.assert_frame_equal(res, expec)
    res = df.query('blob < b', engine=engine, parser=parser)
    expec = df[df.index < df.b]
    tm.assert_frame_equal(res, expec)