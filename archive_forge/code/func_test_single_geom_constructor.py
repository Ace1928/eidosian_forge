import json
import os
import random
import re
import shutil
import tempfile
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_index_equal
from pyproj import CRS
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry
from geopandas import GeoSeries, GeoDataFrame, read_file, datasets, clip
from geopandas._compat import ignore_shapely2_warnings
from geopandas.array import GeometryArray, GeometryDtype
from geopandas.testing import assert_geoseries_equal, geom_almost_equals
from geopandas.tests.util import geom_equals
from pandas.testing import assert_series_equal
import pytest
def test_single_geom_constructor(self):
    p = Point(1, 2)
    line = LineString([(2, 3), (4, 5), (5, 6)])
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], [[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]])
    mp = MultiPoint([(1, 2), (3, 4), (5, 6)])
    mline = MultiLineString([[(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 10)]])
    poly2 = Polygon([(0, 0), (0, -1), (-1, -1), (-1, 0)], [[(-0.1, -0.1), (-0.1, -0.5), (-0.5, -0.5), (-0.5, -0.1)]])
    mpoly = MultiPolygon([poly, poly2])
    geoms = [p, line, poly, mp, mline, mpoly]
    index = ['a', 'b', 'c', 'd']
    for g in geoms:
        gs = GeoSeries(g)
        assert len(gs) == 1
        assert gs.iloc[0].equals(g)
        gs = GeoSeries(g, index=index)
        assert len(gs) == len(index)
        for x in gs:
            assert x.equals(g)