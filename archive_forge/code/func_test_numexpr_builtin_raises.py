from __future__ import annotations
from functools import reduce
from itertools import product
import operator
import numpy as np
import pytest
from pandas.compat import PY312
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import (
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
from pandas.core.computation.expressions import (
from pandas.core.computation.ops import (
from pandas.core.computation.scope import DEFAULT_GLOBALS
def test_numexpr_builtin_raises(engine, parser):
    sin, dotted_line = (1, 2)
    if engine == 'numexpr':
        msg = 'Variables in expression .+'
        with pytest.raises(NumExprClobberingError, match=msg):
            pd.eval('sin + dotted_line', engine=engine, parser=parser)
    else:
        res = pd.eval('sin + dotted_line', engine=engine, parser=parser)
        assert res == sin + dotted_line