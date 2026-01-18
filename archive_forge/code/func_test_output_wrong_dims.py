from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def test_output_wrong_dims() -> None:
    variable = xr.Variable('x', np.arange(10))

    def add_dim(x):
        return x[..., np.newaxis]

    def remove_dim(x):
        return x[..., 0]
    with pytest.raises(ValueError, match='unexpected number of dimensions.*from:\\n\\n.*array\\(\\[\\[0'):
        apply_ufunc(add_dim, variable, output_core_dims=[('y', 'z')])
    with pytest.raises(ValueError, match='unexpected number of dimensions'):
        apply_ufunc(add_dim, variable)
    with pytest.raises(ValueError, match='unexpected number of dimensions'):
        apply_ufunc(remove_dim, variable)