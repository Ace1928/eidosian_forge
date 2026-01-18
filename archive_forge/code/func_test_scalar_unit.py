from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
@requires_cftime
def test_scalar_unit() -> None:
    variable = Variable(('x', 'y'), np.array([[0, 1], [2, 3]]), {'units': np.nan})
    result = coding.times.CFDatetimeCoder().decode(variable)
    assert np.isnan(result.attrs['units'])