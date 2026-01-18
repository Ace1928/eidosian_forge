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
def test_performance_warning_for_poor_alignment(self, engine, parser):
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 10)))
    s = Series(np.random.default_rng(2).standard_normal(10000))
    if engine == 'numexpr':
        seen = PerformanceWarning
    else:
        seen = False
    with tm.assert_produces_warning(seen):
        pd.eval('df + s', engine=engine, parser=parser)
    s = Series(np.random.default_rng(2).standard_normal(1000))
    with tm.assert_produces_warning(False):
        pd.eval('df + s', engine=engine, parser=parser)
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 10000)))
    s = Series(np.random.default_rng(2).standard_normal(10000))
    with tm.assert_produces_warning(False):
        pd.eval('df + s', engine=engine, parser=parser)
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 10)))
    s = Series(np.random.default_rng(2).standard_normal(10000))
    is_python_engine = engine == 'python'
    if not is_python_engine:
        wrn = PerformanceWarning
    else:
        wrn = False
    with tm.assert_produces_warning(wrn) as w:
        pd.eval('df + s', engine=engine, parser=parser)
        if not is_python_engine:
            assert len(w) == 1
            msg = str(w[0].message)
            logged = np.log10(s.size - df.shape[1])
            expected = f"Alignment difference on axis 1 is larger than an order of magnitude on term 'df', by more than {logged:.4g}; performance may suffer."
            assert msg == expected