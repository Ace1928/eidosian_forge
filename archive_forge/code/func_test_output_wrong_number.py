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
def test_output_wrong_number() -> None:
    variable = xr.Variable('x', np.arange(10))

    def identity(x):
        return x

    def tuple3x(x):
        return (x, x, x)
    with pytest.raises(ValueError, match="number of outputs.* Received a <class 'numpy.ndarray'> with 10 elements. Expected a tuple of 2 elements:\\n\\narray\\(\\[0"):
        apply_ufunc(identity, variable, output_core_dims=[(), ()])
    with pytest.raises(ValueError, match='number of outputs'):
        apply_ufunc(tuple3x, variable, output_core_dims=[(), ()])