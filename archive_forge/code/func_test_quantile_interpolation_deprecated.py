from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@pytest.mark.parametrize('method', ['midpoint', 'lower'])
def test_quantile_interpolation_deprecated(self, method) -> None:
    ds = create_test_data(seed=123)
    q = [0.25, 0.5, 0.75]
    with warnings.catch_warnings(record=True) as w:
        ds.quantile(q, interpolation=method)
        assert len(w) == 1
    with warnings.catch_warnings(record=True):
        with pytest.raises(TypeError, match='interpolation and method keywords'):
            ds.quantile(q, method=method, interpolation=method)