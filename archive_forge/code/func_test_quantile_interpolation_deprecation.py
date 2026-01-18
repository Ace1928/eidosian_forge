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
@pytest.mark.parametrize('method', ['midpoint', 'lower'])
def test_quantile_interpolation_deprecation(self, method) -> None:
    v = Variable(['x', 'y'], self.d)
    q = np.array([0.25, 0.5, 0.75])
    with pytest.warns(FutureWarning, match='`interpolation` argument to quantile was renamed to `method`'):
        actual = v.quantile(q, dim='y', interpolation=method)
    expected = v.quantile(q, dim='y', method=method)
    np.testing.assert_allclose(actual.values, expected.values)
    with warnings.catch_warnings(record=True):
        with pytest.raises(TypeError, match='interpolation and method keywords'):
            v.quantile(q, dim='y', interpolation=method, method=method)