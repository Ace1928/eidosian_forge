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
@pytest.mark.parametrize('slc', [slice(0, 0), slice(100, 0), slice(100, 100), slice(100, 100, -1), slice(0, 100, -1)])
def test_not_slice_like_slices(self, slc):
    assert not BlockPlacement(slc).is_slice_like