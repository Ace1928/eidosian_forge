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
def test_persist(self):
    u = self.eager_array
    v = self.lazy_array + 1
    v2, = dask.persist(v)
    assert v is not v2
    assert len(v2.__dask_graph__()) < len(v.__dask_graph__())
    assert v2.__dask_keys__() == v.__dask_keys__()
    assert dask.is_dask_collection(v)
    assert dask.is_dask_collection(v2)
    self.assertLazyAndAllClose(u + 1, v)
    self.assertLazyAndAllClose(u + 1, v2)