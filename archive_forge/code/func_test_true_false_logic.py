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
def test_true_false_logic(self):
    with tm.maybe_produces_warning(DeprecationWarning, PY312, check_stacklevel=False):
        assert pd.eval('not True') == -2
        assert pd.eval('not False') == -1
        assert pd.eval('True and not True') == 0