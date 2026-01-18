import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_undefined_local(self):
    engine, parser = (self.engine, self.parser)
    skip_if_no_pandas_parser(parser)
    df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list('ab'))
    with pytest.raises(UndefinedVariableError, match="local variable 'c' is not defined"):
        df.query('a == @c', engine=engine, parser=parser)