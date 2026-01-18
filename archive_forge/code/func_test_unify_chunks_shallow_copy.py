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
@pytest.mark.parametrize('obj', [make_ds(), make_da()])
@pytest.mark.parametrize('transform', [lambda x: x.compute(), lambda x: x.unify_chunks()])
def test_unify_chunks_shallow_copy(obj, transform):
    obj = transform(obj)
    unified = obj.unify_chunks()
    assert_identical(obj, unified) and obj is not obj.unify_chunks()