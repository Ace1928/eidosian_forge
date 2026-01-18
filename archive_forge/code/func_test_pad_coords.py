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
def test_pad_coords(self) -> None:
    ar = DataArray(np.arange(3 * 4 * 5).reshape(3, 4, 5), [('x', np.arange(3)), ('y', np.arange(4)), ('z', np.arange(5))])
    actual = ar.pad(x=(1, 3), constant_values=1)
    expected = DataArray(np.pad(np.arange(3 * 4 * 5).reshape(3, 4, 5), mode='constant', pad_width=((1, 3), (0, 0), (0, 0)), constant_values=1), [('x', np.pad(np.arange(3).astype(np.float32), mode='constant', pad_width=(1, 3), constant_values=np.nan)), ('y', np.arange(4)), ('z', np.arange(5))])
    assert_identical(actual, expected)