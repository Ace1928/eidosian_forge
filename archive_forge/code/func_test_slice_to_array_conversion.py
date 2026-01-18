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
@pytest.mark.parametrize('slc, arr', [(slice(0, 3), [0, 1, 2]), (slice(0, 0), []), (slice(3, 0), []), (slice(3, 0, -1), [3, 2, 1])])
def test_slice_to_array_conversion(self, slc, arr):
    tm.assert_numpy_array_equal(BlockPlacement(slc).as_array, np.asarray(arr, dtype=np.intp))