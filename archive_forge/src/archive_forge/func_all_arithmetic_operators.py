import operator
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_series_equal
from pandas.tests.extension import base as extension_tests
import shapely.geometry
from shapely.geometry import Point
from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import (
import pytest
@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations

    Adapted to exclude __sub__, as this is implemented as "difference".
    """
    return request.param