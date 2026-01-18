import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_syntax_error(self, engine, parser):
    df = DataFrame({'i': range(10), '+': range(3, 13), 'r': range(4, 14)})
    msg = 'invalid syntax'
    with pytest.raises(SyntaxError, match=msg):
        df.query('i - +', engine=engine, parser=parser)