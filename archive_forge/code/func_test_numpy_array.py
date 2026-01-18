from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
@pytest.mark.parametrize('input_dict,expected', [({0: 0}, np.array([[0]], dtype=np.int64)), ({'a': 'a'}, np.array([['a']], dtype=object)), ({1: 1}, np.array([[1]], dtype=np.int64))])
def test_numpy_array(input_dict, expected):
    result = np.array([Series(input_dict)])
    tm.assert_numpy_array_equal(result, expected)