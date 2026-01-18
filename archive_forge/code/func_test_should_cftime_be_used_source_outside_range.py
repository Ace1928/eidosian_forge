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
def test_should_cftime_be_used_source_outside_range():
    src = cftime_range('1000-01-01', periods=100, freq='MS', calendar='noleap')
    with pytest.raises(ValueError, match='Source time range is not valid for numpy datetimes.'):
        _should_cftime_be_used(src, 'standard', False)