import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_with_nested_strings(self, parser, engine):
    skip_if_no_pandas_parser(parser)
    events = [f'page {n} {act}' for n in range(1, 4) for act in ['load', 'exit']] * 2
    stamps1 = date_range('2014-01-01 0:00:01', freq='30s', periods=6)
    stamps2 = date_range('2014-02-01 1:00:01', freq='30s', periods=6)
    df = DataFrame({'id': np.arange(1, 7).repeat(2), 'event': events, 'timestamp': stamps1.append(stamps2)})
    expected = df[df.event == '"page 1 load"']
    res = df.query('\'"page 1 load"\' in event', parser=parser, engine=engine)
    tm.assert_frame_equal(expected, res)