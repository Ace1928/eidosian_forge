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
@pytest.mark.parametrize('q', [0.25, [0.5], [0.25, 0.75]])
@pytest.mark.parametrize('axis, dim', [[1, 'y'], [[1], ['y']]])
def test_quantile_dask(self, q, axis, dim):
    v = Variable(['x', 'y'], self.d).chunk({'x': 2})
    actual = v.quantile(q, dim=dim)
    assert isinstance(actual.data, dask_array_type)
    expected = np.nanpercentile(self.d, np.array(q) * 100, axis=axis)
    np.testing.assert_allclose(actual.values, expected)