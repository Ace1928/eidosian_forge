import random
import numpy as np
import pandas as pd
from pyproj import CRS
import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE, BaseGeometry
import shapely.wkb
import shapely.wkt
import geopandas
from geopandas.array import (
import geopandas._compat as compat
import pytest
@pytest.mark.parametrize('attr', ['area', 'length'])
def test_unary_float(attr):
    na_value = np.nan
    result = getattr(T, attr)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype('float64')
    expected = [getattr(t, attr) if t is not None else na_value for t in triangles]
    np.testing.assert_allclose(result, expected)