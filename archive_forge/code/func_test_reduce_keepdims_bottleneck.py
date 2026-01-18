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
@requires_bottleneck
def test_reduce_keepdims_bottleneck(self) -> None:
    import bottleneck
    coords = {'x': [-1, -2], 'y': ['ab', 'cd', 'ef'], 'lat': (['x', 'y'], [[1, 2, 3], [-1, -2, -3]]), 'c': -999}
    orig = DataArray([[-1, 0, 1], [-3, 0, 3]], coords, dims=['x', 'y'])
    actual = orig.reduce(bottleneck.nanmean, keepdims=True)
    expected = orig.mean(keepdims=True)
    assert_equal(actual, expected)