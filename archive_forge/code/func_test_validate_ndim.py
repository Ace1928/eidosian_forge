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
def test_validate_ndim():
    values = np.array([1.0, 2.0])
    placement = BlockPlacement(slice(2))
    msg = 'Wrong number of dimensions. values.ndim != ndim \\[1 != 2\\]'
    with pytest.raises(ValueError, match=msg):
        make_block(values, placement, ndim=2)