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
@requires_dask
def test_sparse_dask_dataset_repr(self):
    ds = xr.Dataset(data_vars={'a': ('x', sparse.COO.from_numpy(np.ones(4)))}).chunk()
    expected = dedent('            <xarray.Dataset> Size: 32B\n            Dimensions:  (x: 4)\n            Dimensions without coordinates: x\n            Data variables:\n                a        (x) float64 32B dask.array<chunksize=(4,), meta=sparse.COO>')
    assert expected == repr(ds)