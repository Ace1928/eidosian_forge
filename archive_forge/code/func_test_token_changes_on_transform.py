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
@pytest.mark.parametrize('obj', [make_da(), make_da().compute(), make_ds(), make_ds().compute()])
@pytest.mark.parametrize('transform', [lambda x: x.reset_coords(), lambda x: x.reset_coords(drop=True), lambda x: x.isel(x=1), lambda x: x.attrs.update(new_attrs=1), lambda x: x.assign_coords(cxy=1), lambda x: x.rename({'x': 'xnew'}), lambda x: x.rename({'cxy': 'cxynew'})])
def test_token_changes_on_transform(obj, transform):
    with raise_if_dask_computes():
        assert dask.base.tokenize(obj) != dask.base.tokenize(transform(obj))