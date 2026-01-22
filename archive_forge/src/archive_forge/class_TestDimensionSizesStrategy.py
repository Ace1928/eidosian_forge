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
class TestDimensionSizesStrategy:

    @given(dimension_sizes())
    def test_types(self, dims):
        assert isinstance(dims, dict)
        for d, n in dims.items():
            assert isinstance(d, str)
            assert len(d) >= 1
            assert isinstance(n, int)
            assert n >= 0

    @given(st.data(), st.tuples(st.integers(0, 10), st.integers(0, 10)).map(sorted))
    def test_number_of_dims(self, data, ndims):
        min_dims, max_dims = ndims
        dim_sizes = data.draw(dimension_sizes(min_dims=min_dims, max_dims=max_dims))
        assert isinstance(dim_sizes, dict)
        assert min_dims <= len(dim_sizes) <= max_dims

    @given(st.data())
    def test_restrict_names(self, data):
        capitalized_names = st.text(st.characters(), min_size=1).map(str.upper)
        dim_sizes = data.draw(dimension_sizes(dim_names=capitalized_names))
        for dim in dim_sizes.keys():
            assert dim.upper() == dim