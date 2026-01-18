import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_single_element_booleans(self, parser, engine):
    columns = ('bid', 'bidsize', 'ask', 'asksize')
    data = np.random.default_rng(2).integers(2, size=(1, len(columns))).astype(bool)
    df = DataFrame(data, columns=columns)
    res = df.query('bid & ask', engine=engine, parser=parser)
    expected = df[df.bid & df.ask]
    tm.assert_frame_equal(res, expected)