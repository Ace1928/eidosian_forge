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
class TestReduceND(TestReduce):

    @pytest.mark.parametrize('op', ['idxmin', 'idxmax'])
    @pytest.mark.parametrize('ndim', [3, 5])
    def test_idxminmax_dask(self, op: str, ndim: int) -> None:
        if not has_dask:
            pytest.skip('requires dask')
        ar0_raw = xr.DataArray(np.random.random_sample(size=[10] * ndim), dims=[i for i in 'abcdefghij'[:ndim - 1]] + ['x'], coords={'x': np.arange(10)}, attrs=self.attrs)
        ar0_dsk = ar0_raw.chunk({})
        assert_equal(getattr(ar0_dsk, op)(dim='x'), getattr(ar0_raw, op)(dim='x'))