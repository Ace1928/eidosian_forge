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
class TestEstimateUtmCrs:

    def setup_method(self):
        self.esb = shapely.geometry.Point(-73.9847, 40.7484)
        self.sol = shapely.geometry.Point(-74.0446, 40.6893)
        self.landmarks = from_shapely([self.esb, self.sol], crs='epsg:4326')

    def test_estimate_utm_crs__geographic(self):
        assert self.landmarks.estimate_utm_crs() == CRS('EPSG:32618')
        assert self.landmarks.estimate_utm_crs('NAD83') == CRS('EPSG:26918')

    def test_estimate_utm_crs__projected(self):
        assert self.landmarks.to_crs('EPSG:3857').estimate_utm_crs() == CRS('EPSG:32618')

    def test_estimate_utm_crs__antimeridian(self):
        antimeridian = from_shapely([shapely.geometry.Point(1722483.900174921, 5228058.6143420935), shapely.geometry.Point(4624385.494808555, 8692574.544944234)], crs='EPSG:3851')
        assert antimeridian.estimate_utm_crs() == CRS('EPSG:32760')

    def test_estimate_utm_crs__out_of_bounds(self):
        with pytest.raises(RuntimeError, match='Unable to determine UTM CRS'):
            from_shapely([shapely.geometry.Polygon([(0, 90), (1, 90), (2, 90)])], crs='EPSG:4326').estimate_utm_crs()

    def test_estimate_utm_crs__missing_crs(self):
        with pytest.raises(RuntimeError, match='crs must be set'):
            from_shapely([shapely.geometry.Polygon([(0, 90), (1, 90), (2, 90)])]).estimate_utm_crs()