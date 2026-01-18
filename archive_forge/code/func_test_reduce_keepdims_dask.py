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
@requires_dask
def test_reduce_keepdims_dask(self):
    import dask.array
    v = Variable(['x', 'y'], self.d).chunk()
    actual = v.mean(keepdims=True)
    assert isinstance(actual.data, dask.array.Array)
    expected = Variable(v.dims, np.mean(self.d, keepdims=True))
    assert_identical(actual, expected)
    actual = v.mean(dim='y', keepdims=True)
    assert isinstance(actual.data, dask.array.Array)
    expected = Variable(v.dims, np.mean(self.d, axis=1, keepdims=True))
    assert_identical(actual, expected)