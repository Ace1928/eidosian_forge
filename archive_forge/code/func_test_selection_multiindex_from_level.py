from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_selection_multiindex_from_level(self) -> None:
    da = DataArray([0, 1], dims=['x'], coords={'x': [0, 1], 'y': 'a'})
    db = DataArray([2, 3], dims=['x'], coords={'x': [0, 1], 'y': 'b'})
    data = xr.concat([da, db], dim='x').set_index(xy=['x', 'y'])
    assert data.dims == ('xy',)
    actual = data.sel(y='a')
    expected = data.isel(xy=[0, 1]).unstack('xy').squeeze('y')
    assert_equal(actual, expected)