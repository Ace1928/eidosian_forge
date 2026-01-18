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
@pytest.mark.parametrize('persist', [lambda x: x.persist(), lambda x: dask.persist(x)[0]])
def test_persist_DataArray(persist):
    x = da.arange(10, chunks=(5,))
    y = DataArray(x)
    z = y + 1
    n = len(z.data.dask)
    zz = persist(z)
    assert len(z.data.dask) == n
    assert len(zz.data.dask) == zz.data.npartitions