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
def test_shift_has_crs():
    t = T.copy()
    t.crs = 4326
    assert t.shift(1).crs == t.crs
    assert t.shift(0).crs == t.crs
    assert t.shift(-1).crs == t.crs