from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
def test_timedelta64_conversion(self):
    times = pd.timedelta_range(start=0, periods=3)
    for values, preserve_source in [(times, True), (times.values, True), (times.values.astype('timedelta64[s]'), False), (times.to_pytimedelta(), False)]:
        v = self.cls(['t'], values)
        assert v.dtype == np.dtype('timedelta64[ns]')
        assert_array_equal(v.values, times.values)
        assert v.values.dtype == np.dtype('timedelta64[ns]')
        same_source = source_ndarray(v.values) is source_ndarray(values)
        assert preserve_source == same_source