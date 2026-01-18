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
def test_stack_multi_index(self) -> None:
    midx = pd.MultiIndex.from_product([['a', 'b'], [0, 1]], names=('lvl1', 'lvl2'))
    coords = Coordinates.from_pandas_multiindex(midx, 'x')
    coords['y'] = [0, 1]
    ds = xr.Dataset(data_vars={'b': (('x', 'y'), [[0, 1], [2, 3], [4, 5], [6, 7]])}, coords=coords)
    expected = Dataset(data_vars={'b': ('z', [0, 1, 2, 3, 4, 5, 6, 7])}, coords={'x': ('z', np.repeat(midx.values, 2)), 'lvl1': ('z', np.repeat(midx.get_level_values('lvl1'), 2)), 'lvl2': ('z', np.repeat(midx.get_level_values('lvl2'), 2)), 'y': ('z', [0, 1, 0, 1] * 2)})
    actual = ds.stack(z=['x', 'y'], create_index=False)
    assert_identical(expected, actual)
    assert len(actual.xindexes) == 0
    with pytest.raises(ValueError, match='cannot create.*wraps a multi-index'):
        ds.stack(z=['x', 'y'], create_index=True)