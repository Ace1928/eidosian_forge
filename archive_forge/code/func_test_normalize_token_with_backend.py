from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
@requires_scipy_or_netCDF4
def test_normalize_token_with_backend(map_ds):
    with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as tmp_file:
        map_ds.to_netcdf(tmp_file)
        read = xr.open_dataset(tmp_file)
        assert not dask.base.tokenize(map_ds) == dask.base.tokenize(read)
        read.close()