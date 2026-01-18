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
@given(st.data())
def test_restrict_names(self, data):
    capitalized_names = st.text(st.characters(), min_size=1).map(str.upper)
    dim_sizes = data.draw(dimension_sizes(dim_names=capitalized_names))
    for dim in dim_sizes.keys():
        assert dim.upper() == dim