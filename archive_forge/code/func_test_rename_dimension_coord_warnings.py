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
def test_rename_dimension_coord_warnings(self) -> None:
    ds = Dataset(coords={'x': ('y', [0, 1])})
    with pytest.warns(UserWarning, match="rename 'x' to 'y' does not create an index.*"):
        ds.rename(x='y')
    ds = Dataset(coords={'y': ('x', [0, 1])})
    with pytest.warns(UserWarning, match="rename 'x' to 'y' does not create an index.*"):
        ds.rename(x='y')
    ds = Dataset(data_vars={'data': (('x', 'y'), np.ones((2, 3)))}, coords={'x': range(2), 'y': range(3), 'a': ('x', [3, 4])})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ds.rename(x='x')