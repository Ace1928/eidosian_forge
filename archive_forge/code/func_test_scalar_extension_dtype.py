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
def test_scalar_extension_dtype(self, ea_scalar_and_dtype):
    ea_scalar, ea_dtype = ea_scalar_and_dtype
    ser = Series(ea_scalar, index=range(3))
    expected = Series([ea_scalar] * 3, dtype=ea_dtype)
    assert ser.dtype == ea_dtype
    tm.assert_series_equal(ser, expected)