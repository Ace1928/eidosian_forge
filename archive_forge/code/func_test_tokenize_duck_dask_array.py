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
@requires_pint
def test_tokenize_duck_dask_array(self):
    import pint
    unit_registry = pint.UnitRegistry()
    q = unit_registry.Quantity(self.data, unit_registry.meter)
    data_array = xr.DataArray(data=q, coords={'x': range(4)}, dims=('x', 'y'), name='foo')
    token = dask.base.tokenize(data_array)
    post_op = data_array + 5 * unit_registry.meter
    assert dask.base.tokenize(data_array) != dask.base.tokenize(post_op)
    assert dask.base.tokenize(data_array) == token