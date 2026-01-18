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
def test_to_dask_dataframe_coordinates(self):
    x = np.random.randn(10)
    t = np.arange(10) * 2
    ds = Dataset({'a': ('t', da.from_array(x, chunks=4)), 't': ('t', da.from_array(t, chunks=4))})
    expected_pd = pd.DataFrame({'a': x}, index=pd.Index(t, name='t'))
    expected = dd.from_pandas(expected_pd, chunksize=4)
    actual = ds.to_dask_dataframe(set_index=True)
    assert isinstance(actual, dd.DataFrame)
    assert_frame_equal(expected.compute(), actual.compute())