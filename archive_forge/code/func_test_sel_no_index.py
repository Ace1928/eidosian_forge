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
def test_sel_no_index(self) -> None:
    array = DataArray(np.arange(10), dims='x')
    assert_identical(array[0], array.sel(x=0))
    assert_identical(array[:5], array.sel(x=slice(5)))
    assert_identical(array[[0, -1]], array.sel(x=[0, -1]))
    assert_identical(array[array < 5], array.sel(x=array < 5))