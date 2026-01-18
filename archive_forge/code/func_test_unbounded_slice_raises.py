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
@pytest.mark.parametrize('slc', [slice(None, None), slice(10, None), slice(None, None, -1), slice(None, 10, -1), slice(-1, None), slice(None, -1), slice(-1, -1), slice(-1, None, -1), slice(None, -1, -1), slice(-1, -1, -1)])
def test_unbounded_slice_raises(self, slc):
    msg = 'unbounded slice'
    with pytest.raises(ValueError, match=msg):
        BlockPlacement(slc)