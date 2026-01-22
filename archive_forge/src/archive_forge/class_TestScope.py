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
class TestScope:

    def test_global_scope(self, engine, parser):
        e = '_var_s * 2'
        tm.assert_numpy_array_equal(_var_s * 2, pd.eval(e, engine=engine, parser=parser))

    def test_no_new_locals(self, engine, parser):
        x = 1
        lcls = locals().copy()
        pd.eval('x + 1', local_dict=lcls, engine=engine, parser=parser)
        lcls2 = locals().copy()
        lcls2.pop('lcls')
        assert lcls == lcls2

    def test_no_new_globals(self, engine, parser):
        x = 1
        gbls = globals().copy()
        pd.eval('x + 1', engine=engine, parser=parser)
        gbls2 = globals().copy()
        assert gbls == gbls2

    def test_empty_locals(self, engine, parser):
        x = 1
        msg = "name 'x' is not defined"
        with pytest.raises(UndefinedVariableError, match=msg):
            pd.eval('x + 1', engine=engine, parser=parser, local_dict={})

    def test_empty_globals(self, engine, parser):
        msg = "name '_var_s' is not defined"
        e = '_var_s * 2'
        with pytest.raises(UndefinedVariableError, match=msg):
            pd.eval(e, engine=engine, parser=parser, global_dict={})