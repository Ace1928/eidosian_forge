import numpy as np
import pandas as pd
from holoviews import Tiles
from holoviews.element.comparison import ComparisonTestCase
def test_spot_check_eastingnorthing_to_lonlat(self):
    lon, lat = Tiles.easting_northing_to_lon_lat(0, 0)
    self.assertAlmostEqual(lon, 0)
    self.assertAlmostEqual(lat, 0)
    lon, lat = Tiles.easting_northing_to_lon_lat(1230020, -432501)
    self.assertAlmostEqual(lon, 11.0494578, places=2)
    self.assertAlmostEqual(lat, -3.8822487, places=2)
    lon, lat = Tiles.easting_northing_to_lon_lat(-2130123, 1829312)
    self.assertAlmostEqual(lon, -19.1352205, places=2)
    self.assertAlmostEqual(lat, 16.2122187, places=2)
    lon, lat = Tiles.easting_northing_to_lon_lat(-1000000, 5000000)
    self.assertAlmostEqual(lon, -8.9831528, places=2)
    self.assertAlmostEqual(lat, 40.9162745, places=2)
    lon, lat = Tiles.easting_northing_to_lon_lat(-20037508.34, 20037508.34)
    self.assertAlmostEqual(lon, -180.0, places=2)
    self.assertAlmostEqual(lat, 85.0511288, places=2)