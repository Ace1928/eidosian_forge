import numpy as np
import numpy.testing as npt
import pytest
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.array_api import make_strategies_namespace
from xarray.core.variable import Variable
from xarray.testing.strategies import (
from xarray.tests import requires_numpy_array_api
@given(st.data(), st.tuples(st.integers(0, 10), st.integers(0, 10)).map(sorted))
def test_number_of_dims(self, data, ndims):
    min_dims, max_dims = ndims
    dim_sizes = data.draw(dimension_sizes(min_dims=min_dims, max_dims=max_dims))
    assert isinstance(dim_sizes, dict)
    assert min_dims <= len(dim_sizes) <= max_dims