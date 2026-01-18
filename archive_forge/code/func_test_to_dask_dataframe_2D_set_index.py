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
@pytest.mark.xfail(raises=NotImplementedError)
def test_to_dask_dataframe_2D_set_index(self):
    w = da.from_array(np.random.randn(2, 3), chunks=(1, 2))
    ds = Dataset({'w': (('x', 'y'), w)})
    ds['x'] = ('x', np.array([0, 1], np.int64))
    ds['y'] = ('y', list('abc'))
    expected = ds.compute().to_dataframe()
    actual = ds.to_dask_dataframe(set_index=True)
    assert isinstance(actual, dd.DataFrame)
    assert_frame_equal(expected, actual.compute())