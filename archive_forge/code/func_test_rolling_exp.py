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
@pytest.mark.xfail(reason='Coercion to dense')
def test_rolling_exp(self):
    a1 = self.ds_xr
    a2 = self.sp_xr
    m1 = a1.rolling_exp(x=2, center=True).mean()
    m2 = a2.rolling_exp(x=2, center=True).mean()
    assert isinstance(m2.data, sparse.SparseArray)
    assert np.allclose(m1.data, m2.data.todense())