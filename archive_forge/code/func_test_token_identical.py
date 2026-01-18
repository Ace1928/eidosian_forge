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
@pytest.mark.parametrize('transform', [lambda x: x, lambda x: x.copy(deep=False), lambda x: x.copy(deep=True)])
@pytest.mark.parametrize('obj', [make_da(), make_ds(), make_ds().variables['a']])
def test_token_identical(obj, transform):
    with raise_if_dask_computes():
        assert dask.base.tokenize(obj) == dask.base.tokenize(transform(obj))
    assert dask.base.tokenize(obj.compute()) == dask.base.tokenize(transform(obj.compute()))