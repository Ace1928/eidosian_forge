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
@pytest.mark.parametrize('obj', [make_da().compute(), make_ds().compute()])
def test_token_changes_when_buffer_changes(obj):
    with raise_if_dask_computes():
        t1 = dask.base.tokenize(obj)
    if isinstance(obj, DataArray):
        obj[0, 0] = 123
    else:
        obj['a'][0, 0] = 123
    with raise_if_dask_computes():
        t2 = dask.base.tokenize(obj)
    assert t2 != t1
    obj.coords['ndcoord'][0] = 123
    with raise_if_dask_computes():
        t3 = dask.base.tokenize(obj)
    assert t3 != t2