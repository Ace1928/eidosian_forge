import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_doesnt_pickup_local(self, engine, parser):
    n = m = 10
    df = DataFrame(np.random.default_rng(2).integers(m, size=(n, 3)), columns=list('abc'))
    with pytest.raises(UndefinedVariableError, match="name 'sin' is not defined"):
        df.query('sin > 5', engine=engine, parser=parser)