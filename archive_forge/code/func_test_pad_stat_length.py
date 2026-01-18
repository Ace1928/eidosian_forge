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
@pytest.mark.parametrize('mode', ('minimum', 'maximum', 'mean', 'median'))
@pytest.mark.parametrize('stat_length', (None, 3, (1, 3), {'dim_0': (2, 1), 'dim_2': (4, 2)}))
def test_pad_stat_length(self, mode, stat_length) -> None:
    ar = DataArray(np.arange(3 * 4 * 5).reshape(3, 4, 5))
    actual = ar.pad(dim_0=(1, 3), dim_2=(2, 2), mode=mode, stat_length=stat_length)
    if isinstance(stat_length, dict):
        stat_length = (stat_length['dim_0'], (4, 4), stat_length['dim_2'])
    expected = DataArray(np.pad(np.arange(3 * 4 * 5).reshape(3, 4, 5), pad_width=((1, 3), (0, 0), (2, 2)), mode=mode, stat_length=stat_length))
    assert actual.shape == (7, 4, 9)
    assert_identical(actual, expected)