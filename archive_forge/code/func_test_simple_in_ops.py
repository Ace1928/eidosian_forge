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
def test_simple_in_ops(self, engine, parser):
    if parser != 'python':
        res = pd.eval('1 in [1, 2]', engine=engine, parser=parser)
        assert res
        res = pd.eval('2 in (1, 2)', engine=engine, parser=parser)
        assert res
        res = pd.eval('3 in (1, 2)', engine=engine, parser=parser)
        assert not res
        res = pd.eval('3 not in (1, 2)', engine=engine, parser=parser)
        assert res
        res = pd.eval('[3] not in (1, 2)', engine=engine, parser=parser)
        assert res
        res = pd.eval('[3] in ([3], 2)', engine=engine, parser=parser)
        assert res
        res = pd.eval('[[3]] in [[[3]], 2]', engine=engine, parser=parser)
        assert res
        res = pd.eval('(3,) in [(3,), 2]', engine=engine, parser=parser)
        assert res
        res = pd.eval('(3,) not in [(3,), 2]', engine=engine, parser=parser)
        assert not res
        res = pd.eval('[(3,)] in [[(3,)], 2]', engine=engine, parser=parser)
        assert res
    else:
        msg = "'In' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval('1 in [1, 2]', engine=engine, parser=parser)
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval('2 in (1, 2)', engine=engine, parser=parser)
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval('3 in (1, 2)', engine=engine, parser=parser)
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval('[(3,)] in (1, 2, [(3,)])', engine=engine, parser=parser)
        msg = "'NotIn' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval('3 not in (1, 2)', engine=engine, parser=parser)
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval('[3] not in (1, 2, [[3]])', engine=engine, parser=parser)