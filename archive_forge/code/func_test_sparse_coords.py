from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.xfail(reason='Coercion of coords to dense')
def test_sparse_coords(self):
    xr.DataArray(sparse.COO.from_numpy(np.arange(4)), dims=['x'], coords={'x': sparse.COO.from_numpy([1, 2, 3, 4])})