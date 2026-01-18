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
@requires_iris
def test_prevent_duplicate_coord_names(self) -> None:
    from iris.coords import DimCoord
    from iris.cube import Cube
    longitude = DimCoord([0, 360], standard_name='longitude', var_name='duplicate')
    latitude = DimCoord([-90, 0, 90], standard_name='latitude', var_name='duplicate')
    data = [[0, 0, 0], [0, 0, 0]]
    cube = Cube(data, dim_coords_and_dims=[(longitude, 0), (latitude, 1)])
    with pytest.raises(ValueError):
        xr.DataArray.from_iris(cube)