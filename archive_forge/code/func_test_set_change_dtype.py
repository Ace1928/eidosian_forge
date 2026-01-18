from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
def test_set_change_dtype(self, mgr):
    mgr.insert(len(mgr.items), 'baz', np.zeros(N, dtype=bool))
    mgr.iset(mgr.items.get_loc('baz'), np.repeat('foo', N))
    idx = mgr.items.get_loc('baz')
    assert mgr.iget(idx).dtype == np.object_
    mgr2 = mgr.consolidate()
    mgr2.iset(mgr2.items.get_loc('baz'), np.repeat('foo', N))
    idx = mgr2.items.get_loc('baz')
    assert mgr2.iget(idx).dtype == np.object_
    mgr2.insert(len(mgr2.items), 'quux', np.random.default_rng(2).standard_normal(N).astype(int))
    idx = mgr2.items.get_loc('quux')
    assert mgr2.iget(idx).dtype == np.dtype(int)
    mgr2.iset(mgr2.items.get_loc('quux'), np.random.default_rng(2).standard_normal(N))
    assert mgr2.iget(idx).dtype == np.float64