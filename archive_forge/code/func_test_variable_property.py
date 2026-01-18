from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize('prop', ['chunks', 'data', 'dims', 'dtype', 'encoding', 'imag', 'nbytes', 'ndim', param('values', marks=xfail(reason='Coercion to dense'))])
def test_variable_property(prop):
    var = make_xrvar({'x': 10, 'y': 5})
    getattr(var, prop)